## Como compilar

Só executar o `make`.

```bash
$ make
```

## Como executar

É necessário redirecionar a saída padrão (`stdout`) para um arquivo no formato PPM.
Opcionalmente, é possível definir o número de threads ao definir a variável `OMP_NUM_THREADS`.
Outras informações printadas pelo código são direcionadas à saída de erro (`stderr`).

```bash
$ ./main <width> > image.ppm
# ou
$ OMP_NUM_THREADS=<n> ./main <width> > image.ppm
```
