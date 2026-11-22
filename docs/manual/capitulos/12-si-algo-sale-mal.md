# Si algo sale mal {#si-algo-sale-mal}

Casi todo lo que puede salir mal tiene arreglo desde el propio juego, y casi nada de lo que hagas borra lo que ya tienes. Este capítulo va por lo que suele pasar: la primera construcción que se para, el juego que va lento, el carro que se atasca y los ajustes que se quedan raros.

## La primera construcción se paró

Si cerraste la ventana, apretaste ESC o se apagó la computadora a mitad de la construcción, no perdiste nada. Cada paso terminado deja su marca y cada provincia terminada su archivo. Abre el juego otra vez y sigue sola desde el último paso que acabó; la ventana lo dice: «Si cierras el juego, la próxima vez sigue en el paso donde iba.»

Si lo que pasó fue un error, la ventana lo enseña debajo del título, con la frase «Algo falló:» y el motivo, y añade «Vuelve a abrir el juego: sigue en el paso donde iba.» Casi siempre eso es todo: el error fue de un momento, la provincia a medias quedó marcada y el segundo intento la termina.

Si vuelve a fallar en el mismo paso, borra la carpeta `build` que hay dentro de la carpeta de datos. Ahí están los pasos a medias de las provincias que se estaban construyendo; las que ya terminaron viven en `cuba/` y no se tocan. Al abrir el juego de nuevo rehace solo esas.

Si aun así no pasa, borra la carpeta de datos entera y deja que construya la isla otra vez desde cero. Es lo mismo que la primera vez, con el mismo tiempo. En Linux está en `~/.local/share/contravia` y en Windows en `%LOCALAPPDATA%\Contravia`; el capítulo [Instalar y la primera vez](#instalar) cuenta qué hay dentro.

Tres cosas hacen fallar la construcción:

- Poca memoria libre. No la rompe, pero la vuelve lenta: el juego construye una provincia por cada cuatro núcleos y por cada 4 GB de memoria libre, y con menos baja a una a la vez. Cierra lo que tengas abierto: si tu máquina tiene núcleos de sobra, levantará más de una provincia a la vez.
- Poco disco. La isla ocupa unos 2 GB, y si el disco se llena a mitad de camino la escritura falla y la construcción se para.
- Una tarjeta gráfica sin Vulkan. Ahí no llega a abrirse la ventana. Actualiza el controlador de la tarjeta, o prueba en otra máquina.

## El juego va lento

Lo primero es saber cuánto va lento. Manejando, F1 muestra u oculta un panel en la esquina de arriba a la izquierda con los cuadros por segundo, el cuadro más lento de los últimos tiempos, lo que se está construyendo y cargando, el tráfico y dónde está el carro. Con ese número sabes si el problema es tu máquina o si es el momento.

Un tirón al entrar en un barrio nuevo es normal: el juego arma lo que se ve mientras manejas. La cortina de «CARGANDO…» tapa lo peor al arrancar, y en mitad del paseo el tirón dura un instante.

Lo que más se nota, en orden:

1. El tráfico. En el modo libre, ponlo en «Sin tráfico»: son carros que se manejan solos por toda la provincia, y son lo que más pide.
2. La calidad gráfica. En «Opciones», baja «Calidad gráfica» de Alto a Medio o a Bajo. Bajo está pensado para laptops: dibuja hasta 7 km de distancia, quita las cornisas y los balcones de las fachadas, deja los árboles con menos detalle desde los 35 m y sin hojas finas, pone la hierba más rala y solo hasta 25 m, usa una sola sombra de 80 m, quita el suavizado de bordes, las sombras de las esquinas y el brillo, baja la lluvia a 1000 gotas y el minimapa a 192 píxeles que se redibujan cada ocho cuadros. Apunta a cuadros de 33 ms, unos 30 por segundo; Alto, que es el juego como era, apunta a 20 ms, y Ultra a 16,7 ms, unos 60 por segundo.
3. Lo que se puede apagar sin bajar el resto: «Lluvia en pantalla» en No y «Minimapa» en No. Las dos filas están en «Opciones», y el minimapa se puede volver a poner cuando quieras.

Si quieres tocar una sola cosa, entra en «Ajustar…», debajo de «Calidad gráfica»: ahí está cada ajuste por separado, dieciocho en total, con su nombre y los valores que usan los presets. En cuanto cambias uno, «Calidad gráfica» pasa a «Personalizado». Los que más rinden son «Detalle de la ciudad», «Distancia», «Árboles», «Hierba» y «Sombras». El capítulo [Opciones y calidad gráfica](#opciones-y-calidad) cuenta qué hace cada uno.

## El carro se atascó

Aprieta R. El juego pone el carro parado en la calle más cercana, mirando a lo largo de ella, y el reloj y el paseo siguen como estaban. Funciona desde cualquier sitio: contra una pared, sobre una acera, mirando al mar o en medio de un solar.

Solo hay un caso en que R no hace nada: cuando no hay ninguna calle a menos de 5 km. Pasa mar adentro o en lo más solo de un cayo. Entonces abre la pausa, entra en «Mapa», baja la vista hasta una calle y elige «Manejar aquí».

Si lo que hiciste fue tirarte al mar, no hace falta que aprietes nada. El carro se hunde durante unos dos segundos, con el morro hacia abajo, y a los 2,6 segundos aparece parado en la carretera seca más cercana a menos de 300 m; si no hay ninguna, en la tierra firme más cercana, y siempre mirando hacia dentro, lejos del agua.

## Los ajustes se quedaron raros

Si el juego abre con una calidad que no puedes bajar, con un sonido que no sale o con el minimapa que no aparece, borra el archivo `settings.json` de la carpeta de configuración:

| Sistema | Archivo |
|---|---|
| Linux | `~/.config/contravia/settings.json` |
| Windows | `%APPDATA%\contravia\settings.json` |

La próxima vez que abras el juego, las opciones vuelven a como vienen de fábrica. En esa misma carpeta están tus fotos del Álbum, en `fotos/` con su lista en `coleccion.json`, y tus diseños del Garaje, en `garaje/`; borrar `settings.json` no toca ninguno de los dos. Si quieres guardar una copia de tus fotos o de tus diseños, copia la carpeta `fotos/` con su `coleccion.json`, y la carpeta `garaje/` junto a ella.

## Cuando sale una versión nueva

El juego no se actualiza solo: descargas el archivo nuevo y lo abres. La primera vez de cada versión vuelve a descomprimir el juego y borra el anterior. Si la versión nueva cambia el formato con que se guarda la isla, la ventana lo avisa: en vez de «Construyendo Cuba» dice «Actualizando Cuba» y la vuelve a construir. Tarda lo mismo que la primera vez, se puede parar y seguir igual, y al final tus fotos, tus diseños y tus opciones siguen donde estaban, porque no viven con la isla.

## Dónde mirar cuando falla

El juego no escribe un archivo de registro: lo que sabe lo dice en pantalla.

- En la construcción, la ventana dice qué paso va y, si algo falla, el motivo, con la frase «Algo falló:».
- Manejando, F1 da los números de la máquina: cuadros por segundo, lo que se está cargando y el tráfico.
- En Linux, si abres el juego desde un terminal con `./contravia`, los mensajes del juego salen ahí, incluso los que aparecen antes de que se abra la ventana. Es la mejor forma de ver por qué no arranca.
- En Windows el juego abre sin ventana de terminal, así que si se cierra antes de llegar al menú no deja ningún mensaje. Lo que viste en pantalla es lo que hay.

Si nada de esto lo arregla, cuéntalo en los [issues del proyecto](https://github.com/apiad/contravia/issues) con lo que dijo la pantalla, tu sistema y si pasó en la primera construcción o manejando. Es una demo técnica: los fallos se arreglan más rápido cuando se sabe en qué pantalla y con qué ajustes pasaron.
