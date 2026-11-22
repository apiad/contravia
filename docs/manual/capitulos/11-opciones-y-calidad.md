# Opciones y calidad gráfica {#opciones-y-calidad}

La pantalla «Opciones» se abre desde el menú principal y desde la pausa. Sus filas se recorren con ↑ y ↓, y cada una se cambia con ← y →; ESC guarda lo elegido y vuelve a donde estabas.

## Las filas

| Fila | Qué hace |
|---|---|
| Calidad gráfica | El preset: Bajo, Medio, Alto, Ultra o Personalizado |
| Ajustar… | Abre la pantalla «Calidad gráfica», con cada ajuste por separado |
| Lluvia en pantalla | Sí o No: las gotas que caen alrededor del carro cuando llueve |
| Unidades | km/h o mph en el velocímetro |
| Minimapa | Sí o No: el mapa redondo de arriba a la derecha |
| Ayuda al manejo | Los seis niveles del capítulo de [Manejar](#manejar) |
| Volumen general, Motor, Efectos, Ambiente, Música | De 0 a 10, cada uno por su lado |
| Créditos y licencias | Quién hizo el juego, de dónde salen los datos y con qué licencias se comparte |

La lluvia, las unidades y el minimapa cambian en el acto. Las cinco filas de volumen se cuentan en [Sonido y música](#sonido-y-musica).

## Los cinco presets

La calidad gráfica es una sola palabra que decide a la vez una veintena de ajustes: cuánta ciudad se dibuja, hasta dónde se ve, cuántos árboles y cuánta hierba hay, cómo son las sombras y qué efectos se aplican a la imagen. Se elige en la primera fila de Opciones.

Alto es el juego como era antes de que existieran los presets, y es con el que abre si nunca cambias nada.

| | Bajo | Medio | Alto | Ultra | Cine |
|---|---|---|---|---|---|
| Distancia de la vista | 7 km | 11 km | 15 km | 15 km | 15 km |
| Detalle de la ciudad | 24–72 | 16–64 | 12–48 | 8–32 | 5 |
| Adornos de fachadas | No | Alrededor | Alrededor | Más lejos | Más lejos |
| Árboles | 35 y 90 m | 45 y 120 m | 60 y 160 m | 90 y 240 m | 120 y 320 m |
| Hierba | Dispersa a 25 m | 12 y 30 m | 20 y 45 m | 30 y 70 m | 40 y 90 m |
| Sombras | 80 m, 1 024 | 150 m, 2 048 | 200 m, 2 048 | 250 m, cuatro capas | 300 m, 4 096 |
| Gotas de lluvia | 1 000 | 2 000 | 3 000 | 6 000 | 10 000 |
| Cuadro que persigue | 30 por segundo | 40 | 50 | 60 | No se ajusta |

Los números de «Detalle de la ciudad» son píxeles de error en pantalla: cuanto más bajos, más fina se dibuja la ciudad y más trabajo cuesta. En «Árboles» son las dos distancias a las que un árbol pasa a una forma más simple y luego a la más simple. En «Hierba», la hierba densa llega hasta la primera distancia y los manojos sueltos hasta la segunda. «Gotas de lluvia» son las rayas de agua de la lluvia más fuerte.

Cada paso también cambia cosas que no se ven en una tabla. Bajo apaga el suavizado de bordes, el brillo y las sombras en las esquinas, y deja las sombras en un solo mapa duro de 80 m. Medio enciende el suavizado y un brillo suave. Alto añade las sombras en las esquinas y reparte las sombras en tres capas, nítidas cerca del carro y hasta 200 m. Ultra sube las sombras a cuatro capas de 250 m, añade los rayos de sol, sube las gotas de lluvia a 6 000 y dibuja el minimapa más grande y más seguido. Cine sube el detalle, los árboles, la hierba, las sombras y la lluvia por encima de Ultra, con sombras de 4 096, y no ajusta nada por el tiempo de cuadro.

Cine no está en la lista de Opciones. Es el preset de las capturas: no respeta el tiempo de cuadro y sirve para grabar y fotografiar, no para jugar.

## Cuál elegir

- Una laptop o una tarjeta vieja: Bajo. Dibuja menos ciudad a lo lejos, no pone adornos en las fachadas, apaga los efectos de pantalla y baja el tráfico cercano a 200 m.
- Una tarjeta de gama media: Medio, con los adornos de fachadas y el suavizado de bordes, y algo menos de distancia.
- Una tarjeta buena: Alto, el preset de siempre. Ultra es para tarjetas de gama alta y para mirar La Habana con calma.
- Para una foto o un video: Cine, con `contravia play --quality cine`.

El «Cuadro que persigue» no es una promesa sino una meta. Si la máquina no llega, el juego sube por su cuenta el error de pantalla dentro del rango del preset, es decir, dibuja la ciudad más gruesa, y hasta quita los adornos de fachadas; si aun así va a tirones, baja un preset. Al revés también funciona: cuando sobra tiempo, el juego vuelve a bajar el error y dibuja más detalle.

## Ajustar… y Personalizado

ENTER en «Ajustar…» abre la pantalla «Calidad gráfica», con una fila por cada ajuste: Detalle de la ciudad, Distancia, Niebla, Adornos de fachadas, Textura del suelo, Árboles, Hojas finas, Hierba, Cementerios, Tráfico cercano, Sombras, Gotas de lluvia, Minimapa, Suavizado de bordes, Sombras en las esquinas, Brillo, Rayos de sol y Color. Son dieciocho filas en tres columnas.

Cada fila ofrece solo los valores que usa algún preset, y los nombra con el preset del que salen. Al mover una fila, la calidad pasa a ser «Personalizado» y todo se aplica al momento; si vuelves a dejar todas las filas como estaban, vuelve a ser el preset del que partiste. Personalizado empieza siendo el preset que tenías, y cambia solo lo que tocas.

Casi todo se ve al momento. El tamaño del minimapa se toma la próxima vez que abres el juego, y la textura del suelo llega a las celdas que se construyan de ahí en adelante. La fila «Minimapa» de Opciones, la de Sí o No, sí se ve al momento.

ESC guarda y vuelve a Opciones; ESC otra vez guarda las opciones y vuelve al menú o a la pausa. No hay un botón de cancelar: lo que cambiaste queda.

## Dónde se guardan

Las opciones viven en un archivo `settings.json` en la carpeta de configuración del juego:

- Linux: `~/.config/contravia/settings.json`.
- Windows: `%APPDATA%\contravia\settings.json`.

Si defines la variable `CONTRAVIA_CONFIG`, el juego usa esa carpeta en su lugar. Al lado queda la carpeta `garaje`, con un archivo por cada diseño tuyo (mira [El Garaje](#garaje)).

Para probar un preset en una sola partida, sin cambiar lo guardado, abre el juego con el preset en la línea de comandos: `contravia play --quality bajo`, `--quality ultra`, `--quality cine`. La opción vale para esa partida; la próxima abre con lo que tengas guardado.

<!-- shot: contravia play --screen options --shot opciones.png | La pantalla de Opciones, con sus doce filas -->

<!-- shot: contravia play --screen quality --shot calidad.png | La pantalla Calidad gráfica, con una fila por ajuste -->

<!-- shot: contravia play --quality bajo --shot calle-bajo.png | La esquina del Capitolio en Bajo -->

<!-- shot: contravia play --quality medio --shot calle-medio.png | La misma esquina en Medio -->

<!-- shot: contravia play --quality alto --shot calle-alto.png | La misma esquina en Alto -->

<!-- shot: contravia play --quality ultra --shot calle-ultra.png | La misma esquina en Ultra -->

<!-- shot: contravia play --quality cine --shot calle-cine.png | La misma esquina en Cine -->
