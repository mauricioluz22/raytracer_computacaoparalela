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

static int const TAG_REQUEST = 20;
static int const TAG_WORK = 20;
static int const TAG_DIE = 30;

int main(int argc, char *argv[]) {
    int width = 1200;
    if (argc == 2) {
        width = std::atoi(argv[1]);
    }
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
    int const image_height = cam.get_image_height();
    std::vector<color> pixels;
    pixels.resize(cam.image_width * image_height * 2);
    color *raw_data = pixels.data();
    
    // inicializar MPI por volta daqui
    int myrank, size;
    MPI_Status status;
    int done = 0;
    int line = 0;
    MPI_Init(&argc, &argv);

    MPI_Datatype MPI_VEC3;
    int block_lengths[1] = {3}; // 3 doubles
    MPI_Aint offsets[1] = {offsetof(color, e)};
    MPI_Datatype types[1] = {MPI_DOUBLE};
    // 1 == comprimento dos arrays. em geral, os tres terao o mesmo tamanho
    MPI_Type_create_struct(1, block_lengths, offsets, types, &MPI_VEC3);
    MPI_Type_commit(&MPI_VEC3);

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) {
        for (int line = 0; line < image_height; line++) {
            cam.render_line(world, line, &raw_data[line * cam.image_width]);
        }
        cam.write_image(pixels);
        //cam.render(world);
    } else if (myrank == 0) {
        // Loop:
        // MPI_Probe para verificar se ha mensagem e verificar quem a enviou antes de a receber
        // Se for TAG_REQUEST, enviar uma linha pro trabalhador e salvar linha enviada num dicionario/array
        // Se for TAG_WORK E ainda ha linhas para renderizar, repete instrucao anterior. Ademais, salvar vetor retornado pelo trabalhador
        // Se nao houverem linhas para renderizar, enviar TAG_DIE para todos os trabalhadores
        // Responder o primeiro trabalhador a pedir por mais trabalho, independente do quão frequentemente este o peça

        // coordenador inicializa o arquivo da imagem (evita que o header ppm seja salvo multiplas vezes)
        std::unordered_map<int, int> line_per_worker;
        // cam.write_image(pixels);
        for (int i = 1; i < size; i++) {
            // MPI_Send(&i, 1, MPI_INT, i, TAG_DIE, MPI_COMM_WORLD);
            line_per_worker[i] = line;
            MPI_Send(&line, 1, MPI_INT, i, TAG_WORK, MPI_COMM_WORLD);
            line += 1;
        }
        // contar linhas recebidas dos trabalhadores para saber se há mais trabalho para fazer
        int received = 0;
        while (!done) {
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == TAG_WORK) {
            // std::clog << "trying 0 work" << std::endl;
                // std::clog << line_per_worker[status.MPI_SOURCE] << " " << &raw_data[line_per_worker[status.MPI_SOURCE] * cam.image_width] << std::endl;
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
                if (received >= image_height) {
                    done = 1;
                } else if (line < image_height) {
                    line_per_worker[status.MPI_SOURCE] = line;
                    MPI_Send(&line, 1, MPI_INT, status.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD);
                    line += 1;
                }
            }
        }
        // tarefa do coordenador = enviar número da linha a ser renderizada aos trabalhadores [0, image_height)
        cam.write_image(pixels);
        for (int i = 1; i < size; i++) {
            MPI_Send(&i, 1, MPI_INT, i, TAG_DIE, MPI_COMM_WORLD);
        }
    } else {
        // Loop:
        // MPI_Probe para verificar se ha trabalho para renderizar antes de tentar receber mensagem
        // Se for TAG_WORK, renderizar linha, devolver vetor para o coordenador e pedir por mais trabalho
        // Se for TAG_DIE, finalizar a execucao
        // Enviar TAG_REQUEST ao terminar o trabalho dentro do loop e aguardar resposta do coordenador

        while (!done) {
            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == TAG_DIE) {
                // std::clog << "trying n die" << std::endl;
                int tmp;
                MPI_Recv(&tmp, 1, MPI_INT, 0, status.MPI_TAG, MPI_COMM_WORLD, &status);
                done = 1;
            } else if (status.MPI_TAG == TAG_WORK) {
                // std::clog << "trying n work" << std::endl;
                MPI_Recv(&line, 1, MPI_INT, 0, status.MPI_TAG, MPI_COMM_WORLD, &status);
                cam.render_line(world, line, &raw_data[line * cam.image_width]);
                MPI_Send(&raw_data[line * cam.image_width], cam.image_width, MPI_VEC3, 0, TAG_WORK, MPI_COMM_WORLD);
            }

        }
        // tarefa dos trabalhadores = renderizar linhas requisitadas pelo coordenador
    }
    // std::clog << cam.get_image_height() << std::endl;
    // cam.render(world);
    // terminar MPI exatamente aqui
    MPI_Type_free(&MPI_VEC3);
    MPI_Finalize();
}
