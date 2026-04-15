#ifndef CAMERA_H
#define CAMERA_H
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

#include "hittable.h"
#include "material.h"

#include <omp.h>
#include <cstdlib>
#include <random>

class camera {
  public:
    double aspect_ratio      = 1.0;  // Ratio of image width over height
    int    image_width       = 100;  // Rendered image width in pixel count
    int    samples_per_pixel = 10;   // Count of random samples for each pixel
    int    max_depth         = 10;   // Maximum number of ray bounces into scene

    double vfov     = 90;              // Vertical view angle (field of view)
    point3 lookfrom = point3(0,0,0);   // Point camera is looking from
    point3 lookat   = point3(0,0,-1);  // Point camera is looking at
    vec3   vup      = vec3(0,1,0);     // Camera-relative "up" direction

    double defocus_angle = 0;  // Variation angle of rays through each pixel
    double focus_dist = 10;    // Distance from camera lookfrom point to plane of perfect focus

    // candidato ao openmp
    void render(const hittable& world) {
        initialize();

        std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
        std::vector<color> pixels;
        // pré-alocar vetor. agiliza processamento ao evitar realocações e permitir prefetching.
        pixels.resize(image_width * image_height); 
        int j, i;
        double t1 = omp_get_wtime();
        // collapse para paralelizar ambos os loops.
        // guided parece ter um desempenho melhor conforme o tamanho da imagem cresce (dado o aumento da complexidade)
        #pragma omp parallel for collapse(2) schedule(guided, (image_height * image_width) / omp_get_num_threads()) private(j, i)
        for (j = 0; j < image_height; j++) {
            for (i = 0; i < image_width; i++) {
                color pixel_color(0,0,0);
                int sample;
                // carrega algumas das cores do array na cache antes de seu uso
                // evita cache-misses e evita acessos à memória principal
                if (i + 64 < image_width) {
                    __builtin_prefetch(&pixels[i + 64 + j * image_width], 1, 2);
                }
                for (sample = 0; sample < samples_per_pixel; sample++) {
                    ray r = get_ray(i, j);
                    pixel_color += ray_color(r, max_depth, world);
                }
                // write_color(std::cout, pixel_samples_scale * pixel_color);
                pixels[i + j * image_width] = pixel_samples_scale * pixel_color;
            }
        }
        // std::clog << "Elapsed time: " << (omp_get_wtime() - t1) << std::endl;
        std::clog << (omp_get_wtime() - t1) << std::endl;

        for (j = 0; j < image_height; j++) {
            for (i = 0; i < image_width; i++) {
                write_color(std::cout, pixels[i + j * image_width]);
            }
        }
    }

  private:
    int    image_height;         // Rendered image height
    double pixel_samples_scale;  // Color scale factor for a sum of pixel samples
    point3 center;               // Camera center
    point3 pixel00_loc;          // Location of pixel 0, 0
    vec3   pixel_delta_u;        // Offset to pixel to the right
    vec3   pixel_delta_v;        // Offset to pixel below
    vec3   u, v, w;              // Camera frame basis vectors
    vec3   defocus_disk_u;       // Defocus disk horizontal radius
    vec3   defocus_disk_v;       // Defocus disk vertical radius
    std::vector<std::mt19937> rngs; // geradores de números pseudo aleatórios divididos por thread para evitar concorrência

    void initialize() {
        // hack para pegar numero de threads em codigo sequencial
        int size = 1;
        // shared pois uma mesma variável está sendo modificada em cada thread...
        // provavelmente desnecessário dado que todas retornarão o mesmo valor
        #pragma omp parallel shared(size)
        {
            // OpenMP só retorna o número correto de threads em seções paralelas pelo que vi :thinking: 
            #pragma omp single
            size = omp_get_num_threads();
        }
        rngs.resize(size);
        for (auto& rng : rngs) {
            rng.seed(std::random_device{}());
        }
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        pixel_samples_scale = 1.0 / samples_per_pixel;

        center = lookfrom;

        // Determine viewport dimensions.
        auto theta = degrees_to_radians(vfov);
        auto h = std::tan(theta/2);
        auto viewport_height = 2 * h * focus_dist;
        auto viewport_width = viewport_height * (double(image_width)/image_height);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        vec3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
        vec3 viewport_v = viewport_height * -v;  // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left = center - (focus_dist * w) - viewport_u/2 - viewport_v/2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        // Calculate the camera defocus disk basis vectors.
        auto defocus_radius = focus_dist * std::tan(degrees_to_radians(defocus_angle / 2));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }

    ray get_ray(int i, int j) const {
        // Construct a camera ray originating from the defocus disk and directed at a randomly
        // sampled point around the pixel location i, j.

        auto offset = sample_square();
        auto pixel_sample = pixel00_loc
                          + ((i + offset.x()) * pixel_delta_u)
                          + ((j + offset.y()) * pixel_delta_v);

        auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample();
        auto ray_direction = pixel_sample - ray_origin;

        return ray(ray_origin, ray_direction);  
    }

    vec3 sample_square() const {
        // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
        // código modificado para passar por parâmetro os rngs de cada thread
        return vec3(random_double_th(rngs[omp_get_thread_num()]) - 0.5, random_double_th(rngs[omp_get_thread_num()]) - 0.5, 0);
    }

    vec3 sample_disk(double radius) const {
        // Returns a random point in the unit (radius 0.5) disk centered at the origin.
        return radius * random_in_unit_disk();
    }

    point3 defocus_disk_sample() const {
        // Returns a random point in the camera defocus disk.
        auto p = random_in_unit_disk();
        return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
    }

    color ray_color(const ray& r, int depth, const hittable& world) const {
        // If we've exceeded the ray bounce limit, no more light is gathered.
        if (depth <= 0)
            return color(0,0,0);

        hit_record rec;

        // função HIT parece consumir maior parte do tempo de processamento
        // particularmente nas esferas, embora isso provavelmente se deva a termos
        // somente esferas no cenário
        // 
        // length squared (na esfera) também, mas isso pode ser por causa
        // da quantidade de esferas
        if (world.hit(r, interval(0.001, infinity), rec)) {
            ray scattered;
            color attenuation;
            if (rec.mat->scatter(r, rec, attenuation, scattered))
                return attenuation * ray_color(scattered, depth-1, world);
            return color(0,0,0);
        }

        vec3 unit_direction = unit_vector(r.direction());
        auto a = 0.5*(unit_direction.y() + 1.0);
        return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
    }
};


#endif
