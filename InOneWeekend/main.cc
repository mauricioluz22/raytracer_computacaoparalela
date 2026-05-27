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

#include <vector>
#include <cstdlib>
#include <unordered_map>

#include "rtweekend.h"

#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"
#include "mpi.h"

static int const TAG_REQUEST = 10;
static int const TAG_WORK = 20;
static int const TAG_DIE = 30;

static inline hittable_list init_world(void) {
    hittable_list world;
    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));

    // geracao de mundo
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
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

int main(int argc, char *argv[]) {
    int width = 1200;
    if (argc == 2) {
        width = std::atoi(argv[1]);
    }
    hittable_list world = init_world();
    camera cam = init_cam(width);
    int const image_height = cam.get_image_height();
    // buffer de pixels. inicializado no coordenador e nos trabalhadores.
    std::vector<color> pixels;
    // necessario ter acesso ao buffer interno do vector para poder receber as linhas dos trabalhadores
    color *raw_data;

    // inicializar MPI por volta daqui
    MPI_Init(&argc, &argv);

    // tipo especial, criado para poder transferir a lista de pixels via mensagem mais facilmente
    MPI_Datatype MPI_VEC3;
    int block_lengths[1] = {3}; // 3 doubles
    MPI_Aint offsets[1] = {offsetof(color, e)};
    MPI_Datatype types[1] = {MPI_DOUBLE};
    // 1 == comprimento dos arrays.
    MPI_Type_create_struct(1, block_lengths, offsets, types, &MPI_VEC3);
    MPI_Type_commit(&MPI_VEC3);

    int myrank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int done = 0;
    int line = 0;
    // variavel criada so pra enviar e receber mensagens de tipo request e die
    // o valor desta nunca eh utilizado em momento algum, portanto seu valor pode ser qualquer um
    int tmp;
    MPI_Status status;
    // caso seja alocado somente um processo
    if (size < 2) {
        pixels.resize(cam.image_width * image_height);
        raw_data = pixels.data();
        for (line = 0; line < image_height; line++) {
            cam.render_line(world, line, &raw_data[line * cam.image_width]);
        }
        cam.write_image(pixels);
        //cam.render(world);
    } else if (myrank == 0) {
        // inicializa o buffer com o tamanho total da imagem
        pixels.resize(cam.image_width * image_height);
        raw_data = pixels.data();
        std::unordered_map<int, int> line_per_worker; // dicionario que guarda as linhas sendo atualmente renderizadas pelos trabalhadores
        std::vector<int> worker_balance_stats(size - 1, 0); // guarda quantidade de linhas renderizadas por trabalhador.
        int received = 0; // contar linhas recebidas dos trabalhadores para saber se ha mais trabalho para fazer
        double t1 = MPI_Wtime(); // comeco do trabalho principal
        while (!done) {
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == TAG_WORK) {
                MPI_Recv(
                    &raw_data[line_per_worker[status.MPI_SOURCE] * cam.image_width],
                    cam.image_width,
                    MPI_VEC3,
                    status.MPI_SOURCE,
                    status.MPI_TAG,
                    MPI_COMM_WORLD,
                    &status
                );
                received += 1;
                worker_balance_stats[status.MPI_SOURCE - 1] += 1;
                if (line < image_height) {
                    line_per_worker[status.MPI_SOURCE] = line;
                    MPI_Send(&line, 1, MPI_INT, status.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD);
                    line += 1;
                } else if (received >= image_height) {
                    done = 1;
                }
            } else if (status.MPI_TAG == TAG_REQUEST) {
                MPI_Recv(&tmp, 1, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &status);
                line_per_worker[status.MPI_SOURCE] = line;
                MPI_Send(&line, 1, MPI_INT, status.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD);
                line += 1;
            }
        }
        double t2 = MPI_Wtime(); // fim do trabalho principal
        cam.write_image(pixels);
        double t3 = MPI_Wtime(); // fim do trabalho incluindo escrita da imagem no disco
        for (int i = 1; i < size; i++) {
            // avisa o trabalhador que acabou o trabalho. manda ele terminar sua execucao
            MPI_Send(&i, 1, MPI_INT, i, TAG_DIE, MPI_COMM_WORLD);
        }
        for (int i = 1; i < size; i++) {
            std::clog << "Worker: " << i << " Linhas renderizadas: " << worker_balance_stats[i - 1] << std::endl;
        }
        std::clog << "Tempo de execucao (somente main loop): " << t2 - t1 << std::endl;
        std::clog << "Tempo de execucao (incluindo escrita): " << t3 - t1 << std::endl;
    } else {
        // para economizar memoria, trabalhadores inicializam o buffer somente com o tamanho da linha
        pixels.resize(cam.image_width);
        raw_data = pixels.data();
        // envia requisicao de trabalho pro coordenador. feito somente uma vez para poupar mensagens
        MPI_Send(&tmp, 1, MPI_INT, 0, TAG_REQUEST, MPI_COMM_WORLD);
        while (!done) {
            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == TAG_DIE) {
                MPI_Recv(&tmp, 1, MPI_INT, 0, status.MPI_TAG, MPI_COMM_WORLD, &status);
                done = 1;
            } else if (status.MPI_TAG == TAG_WORK) {
                MPI_Recv(&line, 1, MPI_INT, 0, status.MPI_TAG, MPI_COMM_WORLD, &status);
                cam.render_line(world, line, raw_data);
                MPI_Send(raw_data, cam.image_width, MPI_VEC3, 0, TAG_WORK, MPI_COMM_WORLD);
            }

        }
    }
    // terminar MPI exatamente aqui
    MPI_Type_free(&MPI_VEC3);
    MPI_Finalize();
}
