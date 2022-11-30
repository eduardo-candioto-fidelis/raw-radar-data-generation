# Mudanças

## GEN_Cube.py

- [x] **{Herick}** Organizar os cubos: Pegar somente 1 frame para processar e não armazenar todos os frames.
- [x] **{Herick}** Verificar o _Mean Channels Filter_ no metodo ``` Velocity() ```.
- [x] **{Herick}** Normalização das imagens.
- [x] **{Herick}** Verificar o _Mean chirps filter_ no método ```Azimuth()```.
- [x] **{Herick}** Resolver espelhamento da imagem RangeAzimuth.
- [x] **{Herick}** Ajustar Plot Range azimuth.
- [x] **{Herick}** Implementar ```__deltaRA()```, resolução azimuth.
- [ ] Verificar Resolução de distancia (**Range-Channel**, **Range-Doppler** e **Range-Azimuth**), de acordo com os dados fornecidos pelo Diogo

## plot.py

- [ ]  Reformular a get attributes.
- [x] **{Herick}** Reformular a CFAR2D.
- [ ] **{Desistido}** Adicionar mais figuaras no plot

## Record.py

- [ ] **{Vinícius}** Gravação de arrays numpy em arquivos binários.
- [ ] **{Vinícius}** Leitura de arrays numpy de arquivos binários.

## CFAR_Lib.py

- [x] **{Luiz}** Instancias das classes a todo momento, somente uma instacia e set de parametros atraves de um método. (CV, CA, OS, 2D)
- [x] **{Luiz}** Otimizar funções de CFAR em python puro.

## Radar_data_viz.py

- [ ] **{Herick executando}** Criar, Organizar e gerar Lib dos plots.
- [ ] Alterar o nome das variaveis para mnemonicos.
- [ ] Organizar a mainloop.
