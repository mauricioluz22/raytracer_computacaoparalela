//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================
//
// VERSAO HIBRIDA MPI + OpenMP
// ---------------------------
// - MPI distribui BLOCOS DE LINHAS (chunks) entre os nos (1 processo pesado por no).
// - Dentro de cada no, OpenMP renderiza as linhas do chunk em modelo workpool
//   (schedule(dynamic), todas as threads trabalham, controle pela estrutura do OpenMP).
//
// Modos do coordenador (rank 0), controlados por variavel de ambiente COORD_MODE:
//   COORD_MODE=worker (padrao) -> rank 0 coordena E renderiza (intercala as duas tarefas).
//   COORD_MODE=dedicated       -> rank 0 SO coordena (nao renderiza nenhum chunk).
// Isso permite a analise pedida: dedicar ou nao um no/nucleo ao coordenador.
//
// Tamanho do chunk controlado por CHUNK_LINES (padrao 16).
//==============================================================================================

#include <vector>
#include <cstdlib>
#include <unordered_map>
#include <string>

#include "rtweekend.h"

#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"
#include "mpi.h"

#ifdef _OPENMP
#include <omp.h>
#endif

static int const TAG_REQUEST = 10;
static int const TAG_WORK    = 20;

// Mensagem de atribuicao de trabalho: linha inicial + numero de linhas do bloco.
struct WorkAssignment {
    int start_line;
    int n_lines;
};

static inline hittable_list init_world(void) {
    // Semente fixa para que todos os processos MPI gerem o mesmo mundo.
    // random_double() usa semente aleatória de hardware (std::random_device),
    // o que causaria mundos diferentes em cada processo e imagem "picotada".
    std::mt19937 scene_rng(42);
    std::uniform_real_distribution<double> scene_dist(0.0, 1.0);
    auto srand_d = [&]() -> double { return scene_dist(scene_rng); };
    auto srand_r = [&](double lo, double hi) -> double { return lo + (hi - lo) * scene_dist(scene_rng); };

    hittable_list world;
    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));

    // geracao de mundo
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = srand_d();
            point3 center(a + 0.9*srand_d(), 0.2, b + 0.9*srand_d());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color(srand_d(), srand_d(), srand_d())
                                * color(srand_d(), srand_d(), srand_d());
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color(srand_r(0.5,1), srand_r(0.5,1), srand_r(0.5,1));
                    auto fuzz = srand_r(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));
    return world;
}

static inline camera init_cam(int width) {
    camera cam;
    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = width;
    cam.samples_per_pixel = 10;
    cam.max_depth         = 20;

    cam.vfov     = 20;
    cam.lookfrom = point3(13,2,3);
    cam.lookat   = point3(0,0,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0.6;
    cam.focus_dist    = 10.0;

    cam.initialize();
    return cam;
}

// Le inteiro de variavel de ambiente, com valor padrao.
static int env_int(const char *name, int def) {
    const char *v = std::getenv(name);
    if (!v) return def;
    int r = std::atoi(v);
    return (r > 0) ? r : def;
}

int main(int argc, char *argv[]) {
    int width = 1200;
    if (argc == 2) {
        width = std::atoi(argv[1]);
    }
    hittable_list world = init_world();
    camera cam = init_cam(width);
    int const image_height = cam.get_image_height();
    int const image_width  = cam.image_width;

    // Configuracoes via ambiente.
    int CHUNK_LINES = env_int("CHUNK_LINES", 16);
    const char *coord_mode_env = std::getenv("COORD_MODE");
    // coord_dedicated = true  -> rank 0 so coordena (nao renderiza).
    // coord_dedicated = false -> rank 0 coordena E renderiza (hibrido no proprio no).
    bool coord_dedicated = (coord_mode_env && std::string(coord_mode_env) == "dedicated");

    MPI_Init(&argc, &argv);

    // Tipo MPI para transferir color (vec3 = 3 doubles).
    MPI_Datatype MPI_VEC3;
    int block_lengths[1] = {3};
    MPI_Aint offsets[1] = {offsetof(color, e)};
    MPI_Datatype types[1] = {MPI_DOUBLE};
    MPI_Type_create_struct(1, block_lengths, offsets, types, &MPI_VEC3);
    MPI_Type_commit(&MPI_VEC3);

    int myrank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Status status;

    if (size < 2) {
        std::vector<color> pixels((size_t)image_width * image_height);
        double t1 = MPI_Wtime();
        cam.render_chunk(world, 0, image_height, pixels.data());
        double t2 = MPI_Wtime();
        cam.write_image(pixels);
        std::clog << "Tempo de execucao (somente render): " << t2 - t1 << std::endl;
        MPI_Type_free(&MPI_VEC3);
        MPI_Finalize();
        return 0;
    }

    if (myrank == 0) {
        // Buffer global: apenas o coordenador aloca a imagem inteira.
        std::vector<color> pixels((size_t)image_width * image_height);
        color *raw_data = pixels.data();

        std::unordered_map<int, WorkAssignment> assignment;
        std::vector<int> worker_balance_stats(size, 0);

        int next_line = 0;
        int received_lines = 0;
        int active_workers = size - 1;

        std::vector<color> coord_buf;
        if (!coord_dedicated) coord_buf.resize((size_t)CHUNK_LINES * image_width);

        double t1 = MPI_Wtime();

        auto take_next_chunk = [&](WorkAssignment &wa) {
            if (next_line >= image_height) { wa.start_line = 0; wa.n_lines = 0; return; }
            wa.start_line = next_line;
            wa.n_lines = std::min(CHUNK_LINES, image_height - next_line);
            next_line += wa.n_lines;
        };

        // Envia proximo chunk ao worker src; n_lines==0 sinaliza termino.
        auto dispatch_next = [&](int src) {
            WorkAssignment nx;
            take_next_chunk(nx);
            if (nx.n_lines > 0)
                assignment[src] = nx;
            else
                active_workers -= 1;
            MPI_Send(&nx, 2, MPI_INT, src, TAG_WORK, MPI_COMM_WORLD);
        };

        while (received_lines < image_height || active_workers > 0) {
            int flag = 0;
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);

            if (!flag) {
                if (!coord_dedicated && next_line < image_height) {
                    WorkAssignment wa;
                    take_next_chunk(wa);
                    if (wa.n_lines > 0) {
                        cam.render_chunk(world, wa.start_line, wa.n_lines, coord_buf.data());
                        std::copy(coord_buf.data(),
                                  coord_buf.data() + (size_t)wa.n_lines * image_width,
                                  &raw_data[(size_t)wa.start_line * image_width]);
                        received_lines += wa.n_lines;
                        worker_balance_stats[0] += wa.n_lines;
                    }
                    continue;
                }
                MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            }

            int src = status.MPI_SOURCE;

            if (status.MPI_TAG == TAG_WORK) {
                WorkAssignment wa = assignment[src];
                MPI_Recv(
                    &raw_data[(size_t)wa.start_line * image_width],
                    wa.n_lines * image_width,
                    MPI_VEC3, src, TAG_WORK, MPI_COMM_WORLD, &status
                );
                received_lines += wa.n_lines;
                worker_balance_stats[src] += wa.n_lines;
                dispatch_next(src);
            } else if (status.MPI_TAG == TAG_REQUEST) {
                int tmp;
                MPI_Recv(&tmp, 1, MPI_INT, src, TAG_REQUEST, MPI_COMM_WORLD, &status);
                dispatch_next(src);
            }
        }

        double t2 = MPI_Wtime();
        cam.write_image(pixels);
        double t3 = MPI_Wtime();

        for (int i = 0; i < size; i++) {
            std::clog << "Worker: " << i << " Linhas renderizadas: "
                      << worker_balance_stats[i] << std::endl;
        }
        std::clog << "Modo coordenador: "
                  << (coord_dedicated ? "dedicated" : "worker")
                  << " | CHUNK_LINES=" << CHUNK_LINES << std::endl;
        std::clog << "Tempo de execucao (somente main loop): " << t2 - t1 << std::endl;
        std::clog << "Tempo de execucao (incluindo escrita): " << t3 - t1 << std::endl;
    } else {
        // Workers alocam apenas um chunk, nunca a imagem inteira.
        std::vector<color> chunk_buf((size_t)CHUNK_LINES * image_width);
        color *raw_data = chunk_buf.data();

        int tmp = 0;
        MPI_Send(&tmp, 1, MPI_INT, 0, TAG_REQUEST, MPI_COMM_WORLD);

        WorkAssignment wa;
        for (;;) {
            MPI_Recv(&wa, 2, MPI_INT, 0, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (wa.n_lines == 0) break;
            cam.render_chunk(world, wa.start_line, wa.n_lines, raw_data);
            MPI_Send(raw_data, wa.n_lines * image_width, MPI_VEC3, 0, TAG_WORK, MPI_COMM_WORLD);
        }
    }

    MPI_Type_free(&MPI_VEC3);
    MPI_Finalize();
    return 0;
}
