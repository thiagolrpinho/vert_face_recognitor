# vert_face_recognitor
## Como
O algoritmo primeiro converte as imagens de uma database conhecida em códigos de incorporação
que depois podem ser usados para comparar com novos rostos. As imagens dessa database devem ser
organizadas em pasta com o nome(ou identificador) de uma pessoa e os nomes das imagens devem
ser compostos pelo nome da pessoa(ou identificador) + um número e uma extensão.

Depois ele compara imagens de outra database no mesmo formato e imprime no terminal se elas
pertencem a pessoa com o mesmo nome(ou identificador) na database conhecida.
### Atualizando a database
Para criar ou atualizar uma database de pessoas conhecidas basta digitar:
```
python atualiza_conhecidos.py
```
Esses dados ficam encriptografados no arquivo "encondings.pickle"

Para especificar uma outra pasta para ser usada como database conhecida, especificar na chamada do terminal:
 --database "CAMINHO_RELATIVO_PASTA"
Para melhorar a perfomance para sistemas com menor poder de processamento, pode-se também usar o argumento
 --detection-method "hog"
## Comparando Pessoas
Para comparar pessoas de uma database incerta, basta digitar:
```
python compara_desconhecidos.py
```
Para especificar uma outra pasta para ser usada como database incerta, especificar na chamada do terminal:
 --database "CAMINHO_RELATIVO_PASTA"
Também pode ser usado o argumento --detection-method para melhorar o desempenho.

