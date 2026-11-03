INTERPRETAÇÃO DE base_variance E alpha_variance_modifier
========================================================

1. O que os dois parâmetros controlam
-------------------------------------

Na Eq. (11) (fitness.py, gaussian_influence), cada célula do mapa vira uma
gaussiana com peso phi~ = phi * energia. A variância dela é:

    sigma_k^2 = sigma_0^2 * (1 + alpha * I_k / max(I))

- base_variance = sigma_0^2, em células^2 (1 célula = 20 m). É a largura
  MÍNIMA da gaussiana, a de uma célula com incerteza zero.
- alpha_variance_modifier = alpha diz QUANTO A INCERTEZA ALARGA A GAUSSIANA.
  A célula mais incerta do mapa chega a sigma_0^2 * (1 + alpha).

O reward de uma célula-alvo é a soma das gaussianas de todas as fontes. O drone
só escolhe destinos numa janela de 20x20 células (+-200 m) em volta dele, mas o
reward dentro da janela recebe contribuições do mapa todo. A variância decide
DE QUE DISTÂNCIA UMA CÉLULA AINDA "PUXA" O DRONE.


2. base_variance (sigma_0^2)
----------------------------

Baixo (perto de 0.25, sigma ~ 0.5 célula):
    O kernel vira quase um delta: o reward de uma célula é praticamente o
    próprio phi~. A escolha é gulosa, célula a célula, e a vizinhança é
    ignorada. Um ponto isolado de alta incerteza atrai tanto quanto uma região
    inteira.

Alto (perto de 100, sigma ~ 10 células ~ 200 m):
    Até as células já conhecidas espalham influência por ~10 células. O mapa
    de reward fica suave, e o drone prefere REGIÕES com muita incerteza
    acumulada a picos isolados. Detalhes finos se perdem: dois alvos próximos
    viram uma "mancha" só.


3. alpha_variance_modifier (alpha)
----------------------------------

alpha = 0 / baixo:
    Todas as gaussianas têm a mesma largura. A incerteza só entra pelo peso
    phi~, e o alcance não depende de quão desconhecida a célula é.

alpha alto:
    Assimetria: células incertas (não visitadas há tempo) têm alcance longo, e
    células recém-visitadas ficam com alcance curto (sigma_0). Áreas
    inexploradas "irradiam" atração de longe, mesmo de fora da janela de
    20x20, e puxam o drone na direção delas. Áreas recém-cobertas só
    influenciam localmente.

Um detalhe que reforça isso: as gaussianas NÃO são normalizadas (não há o
fator 1/(2*pi*sigma^2)), e o termo full/den só corrige bordas e células
excluídas. A massa total de uma fonte cresce com ~sigma^2. Por isso alpha alto
faz as células incertas pesarem mais na soma, além de chegarem mais longe.


4. O que o GA encontrou (ga_logbook_2000.txt, geração 20)
---------------------------------------------------------

- sigma_0^2 ~ 96.5, ou seja sigma_0 ~ 9.8 células ~ 196 m, praticamente no
  limite superior do intervalo (100).
- alpha ~ 92, então sigma_max^2 ~ 96.5 * 93 ~ 8990, ou seja
  sigma_max ~ 95 células ~ 1.9 km. Isso é o tamanho do mapa (100x100). O raio
  de truncamento 3*sigma (~284 células) passa da diagonal (~141), então o
  código cai sempre no ramo sem truncamento.

Na prática: as células mais incertas influenciam O MAPA INTEIRO, e as
recém-visitadas, ~10 células. O GA preferiu um campo de atração global e
suave. Cada drone é puxado para onde está a massa de incerteza, mesmo longe, e
não para o melhor pico local. Isso combina com o termo de incerteza dominando
o custo (248k de 277k).


5. Ressalvas para a tese
------------------------

1. sigma_0^2 ficou colado no limite superior (100). Talvez o ótimo esteja fora
   do intervalo, então vale ampliar o limite ou pelo menos comentar isso.
2. Com sigma_max já do tamanho do mapa, aumentar alpha ainda mais muda pouco o
   alcance e muda principalmente a massa relativa das fontes incertas. Os dois
   parâmetros devem estar bem correlacionados nessa região do espaço, o que
   costuma deixar o GA "deslizar" ao longo de uma crista.
