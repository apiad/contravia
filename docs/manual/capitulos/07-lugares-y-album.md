# Lugares y el Álbum {#lugares-y-album}

La isla tiene 750 lugares para fotografiar, repartidos por las 16 provincias. Son edificios, plazas, monumentos, playas, faros e ingenios, entre otros. Cada uno tiene una ficha con lo que es y un texto que cuenta su historia. Fotografiarlos llena el Álbum.

## Encontrar un lugar

Mientras manejas, el juego busca dos veces por segundo el lugar sin foto más cercano que tengas a la vista. Cuando lo encuentra, abajo en el centro de la pantalla aparece su nombre, la tecla F y la palabra «Foto».

El aviso solo sale para lugares que todavía no tienen foto. Aparece cuando pasas lo bastante cerca y el lugar no queda tapado por un edificio ni por una pared. Cuanto más grande es el lugar, desde más lejos te avisa, pero nunca desde más de 300 m. Si hay varios a la vista, el aviso es para el más cercano.

El minimapa marca los lugares de alrededor con el mismo icono, hasta los doce más cercanos. Están en gris hasta que los fotografías y en amarillo después. En el mapa grande, por debajo de 13 km de altura, aparecen hasta ochenta, con el mismo código de color.

## Apuntar la cámara

Con el aviso en pantalla, aprieta F. El carro se detiene, el reloj se para y la cámara se pone 1,5 m sobre el carro, apuntada al lugar, con el zoom justo para que el lugar ocupe como el 60 % del cuadro.

Arriba a la izquierda aparece el nombre del lugar. Abajo están las teclas:

| Tecla | Qué hace |
|---|---|
| ← → ↑ ↓ | Gira la cámara un poco: hasta 25° a cada lado y 15° arriba o abajo |
| W y S | Acerca y aleja, entre uno y tres aumentos |
| ESPACIO | Dispara la foto |
| ESC | Vuelve a manejar sin foto |

<!-- shot: contravia play --place L0333 --screen aim --shot lugares-apuntar.png | La cámara apuntando al Capitolio para tomarle la foto -->

## Encuadrar y disparar

Aprieta ESPACIO cuando el lugar esté bien encuadrado. El juego mide el cuadro y decide:

- El lugar debe ocupar entre el 30 % y el 90 % de la imagen. Si ocupa menos, está muy lejos; si ocupa más, estás muy cerca.
- Su centro debe caer en el 60 % central del cuadro. Si no, hay que centrarlo.
- Al menos el 80 % del lugar debe quedar dentro de la imagen.
- No puede haber una pared o un edificio entre la cámara y el lugar.

Si algo falla, la nota dice por qué: «Muy lejos», «Muy cerca», «Céntralo» o «Algo lo tapa». Corrige con las flechas o con W y S, y vuelve a disparar.

Cuando la foto está bien, suena el disparador, el juego guarda la imagen y enseña la ficha del lugar.

## La ficha

La ficha de una foto recién tomada lleva la imagen y una etiqueta con «NUEVO · N de M en PROVINCIA». Debajo van el título del lugar, una línea con el tipo, el municipio, el día y la hora del juego y el clima, y el texto que cuenta su historia. ENTER, «Seguir», te devuelve al carro.

La misma ficha se abre desde el mapa o desde el Álbum. Si el lugar ya tiene foto, enseña la foto y el momento del juego en que la tomaste. Si no la tiene, enseña un recuadro gris en su lugar, pero puedes leer igual su texto. ENTER o ESC te devuelve a donde estabas.

<!-- shot: contravia play --place L0333 --screen card --shot lugares-ficha.png | La ficha del Capitolio después de la foto -->

## El Álbum

«Álbum» está en la pausa y en el menú principal. Abre una pantalla con una tarjeta por provincia: el nombre, cuántos lugares llevas de cuántos hay, y una barra de progreso. Las provincias se reparten en cuatro columnas.

| Control | Qué hace |
|---|---|
| ← → ↑ ↓ | Escoge una provincia |
| ENTER | Abre la página de esa provincia |
| ESC | Vuelve a la pausa o al menú principal |

El panel de la derecha lleva el total, «N de 750 lugares», y el conteo por tipo: Iglesias, Fortalezas, Playas y agua, Museos y edificios, Monumentos, Plazas y parques, Teatros y estadios, Estaciones y obras, Naturaleza y Otros. Cada tipo lleva su icono y sus dos números.

La Habana es la provincia con más lugares, 209. La Isla de la Juventud es la que menos tiene, 8.

La página de una provincia enseña sus lugares en una cuadrícula de quince por página. Primero van los que ya tienen foto y después los demás, por título. Cada uno lleva su foto o, si falta, un recuadro gris con el icono, el tipo y el municipio. Debajo va el título. Arriba se lee «N / M · página X de Y».

| Control | Qué hace |
|---|---|
| ← → ↑ ↓ | Escoge un lugar |
| ENTER | Abre su ficha |
| ESC | Vuelve al Álbum |

Si una provincia tiene más de quince lugares, las flechas pasan de página: bajando desde la última fila vas a la siguiente, y subiendo desde la primera vuelves a la anterior.

<!-- shot: contravia play --screen album --shot lugares-album.png | El Álbum con las provincias, el total y el conteo por tipo -->

<!-- shot: contravia play --place L0333 --screen province --shot lugares-provincia.png | La página de La Habana en el Álbum -->

## Volver a un lugar

El Álbum sirve para ver lo que llevas y lo que te falta. Para ir a un lugar, usa el mapa. Abre la pausa con ESC, elige «Mapa» y pon la mira sobre el lugar. ENTER te ofrece «Ver ficha» y «Manejar aquí». También te ofrece «Ir con GPS», y «Quitar ruta» cuando hay una ruta puesta.

«Manejar aquí» te deja en una calle desde la que el lugar se ve bien. «Ir con GPS» te traza la ruta hasta él y te avisa cuando llegas. [El mapa y el GPS](#mapa-y-gps) cuenta las dos cosas con detalle.

## Las fotos en el disco

Las fotos y la colección viven en la carpeta de configuración del juego: en Linux, `~/.config/contravia`; en Windows, `%APPDATA%\contravia`. Dentro, la carpeta `fotos` guarda un PNG por lugar. El archivo `coleccion.json` apunta qué lugares tienes, con la fecha real, el día y la hora del juego y el clima de cada foto.

Si borras esa carpeta, pierdes las fotos y la colección.

## Modo foto

En la pausa hay también un «Modo foto» que no tiene que ver con el Álbum. Ahí la cámara se suelta del carro y vuela a su alrededor: las flechas la giran y W y S la acercan o la alejan. Sirve para mirar el paisaje. ESC te devuelve a la pausa y no se guarda nada.
