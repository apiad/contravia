# Instalar y la primera vez {#instalar}

Contravía no se instala: es un solo archivo que abres, y la primera vez deja la isla construida en tu computadora. Este capítulo dice qué necesita la máquina, qué hace el juego mientras construye, dónde deja lo que construyó y cómo seguir si lo cierras a mitad de camino.

## Lo que necesitas

- Linux o Windows de 64 bits. Por ahora no hay versión para Mac.
- Una tarjeta gráfica con Vulkan.
- Unos 4 GB de memoria libres mientras se construye la isla.
- Unos 2 GB de disco para la isla construida.

## Descargar

Descarga el archivo de tu sistema desde la [última versión en GitHub](https://github.com/apiad/contravia/releases/latest) o desde los botones de la [página del juego](https://apiad.github.io/contravia/). Es el juego entero, con los datos de las 16 provincias adentro, y no hay que instalarlo: donde lo guardes, ahí está. Guarda el archivo donde lo vayas a encontrar, porque es el juego: cada vez que lo abres, él mismo arranca lo que descomprimió.

En Linux hay que darle permiso de ejecución la primera vez:

```bash
chmod +x contravia
./contravia
```

En Windows se abre con doble clic. Si Windows avisa que el programa no está firmado, toca «Más información» y luego «Ejecutar de todas formas».

## La primera vez son dos trabajos

El primero es descomprimir el juego. El archivo trae dentro el motor y el juego; la primera vez que abres una versión, el lanzador los descomprime en tu carpeta de datos y escribe «Contravía: preparando el juego por primera vez...». Tarda poco. Cuando sale una versión nueva, la descomprime otra vez y borra la anterior, para no quedarse con las dos.

El segundo es construir la isla, y es el que tarda. El juego no trae la isla hecha: trae los datos de las 16 provincias, con sus mapas, su relieve y su cobertura del suelo, y arma la isla en tu máquina. Eso se hace una sola vez.

## La ventana «Construyendo Cuba»

Abre una ventana propia, con el logo y la frase «Tu computadora arma la ciudad a partir del mapa: calles, puentes, edificios y aceras. Se hace una sola vez; después juegas sin esperar.» Debajo van los cuatro pasos de cada provincia, con una marca cuando terminan y el porcentaje del que está en curso:

| Paso | Qué hace |
|---|---|
| Leer el mapa | Abre los datos empaquetados de la provincia |
| Edificios | Levanta las casas, los solares vacíos y los parques |
| Calles y puentes | Teje la superficie de las calles, con sus esquinas, sus puentes y sus túneles |
| Aceras | Pone las aceras, los bordes y las entradas de cada edificio |

Las provincias se construyen en orden, empezando por La Habana, y la ventana va diciendo cuál va y cuántas faltan, por ejemplo «LA HABANA · 1 DE 16 · 7 % EN TOTAL», con la barra de la isla entera y el tiempo que queda. Al terminar las 16 viene un último trabajo, «Resumen de la isla»: el juego arma con qué se ve la isla desde lejos en el mapa y la red de rutas del GPS.

Cuánto tarda depende de la máquina. En una de 12 núcleos con 4 GB libres tardó unos 50 minutos, porque construye una provincia a la vez. El juego construye una provincia por cada cuatro núcleos y por cada 4 GB de memoria libre: con 12 GB libres levanta tres a la vez y baja a unos 20 minutos, y en una laptop con menos memoria puede tardar bastante más que la hora.

## Si lo cierras a mitad de camino

Puedes cerrar la ventana, apretar ESC y «Pausar y salir», o apagar la computadora: no pierdes lo hecho. Cada paso que termina deja su marca y cada provincia terminada deja su archivo, y la próxima vez el juego sigue desde el último paso que acabó. La ventana lo dice mientras construye: «Si cierras el juego, la próxima vez sigue en el paso donde iba.»

Si algo falla, la ventana lo dice arriba, con la frase «Algo falló:» y el motivo, y recomienda volver a abrir el juego. El capítulo [Si algo sale mal](#si-algo-sale-mal) cuenta qué hacer si vuelve a fallar en el mismo paso.

Cuando la isla está lista, la ventana cambia de título a «CUBA ESTÁ LISTA» y dice cuánto tardó y cuánto ocupa en disco, con la frase «No hace falta repetirlo». Abajo aparece «JUGAR ›»: aprieta ENTER y el juego abre en el menú principal. Las veces siguientes arranca directo ahí, sin ventana de construcción.

<!-- shot: contravia build --window --shot | La ventana «Construyendo Cuba», con sus cuatro pasos y la barra de la isla -->

## Dónde queda lo que construyó

Todo va a la carpeta de datos de tu usuario, que el juego crea sola:

| Sistema | Carpeta |
|---|---|
| Linux | `~/.local/share/contravia` (o `$XDG_DATA_HOME/contravia`) |
| Windows | `%LOCALAPPDATA%\Contravia` |

Ahí dentro, `runtime/` guarda el juego descomprimido y `cuba/` la isla:

- un archivo por provincia, con el nombre en minúsculas y guiones: `la-habana.npz`, `artemisa.npz`, `pinar-del-rio.npz`, hasta `isla-de-la-juventud.npz`;
- `index.npz`, el índice de la isla, y la carpeta `summary/`, con lo que se ve desde lejos en el mapa;
- `route.npz`, la red sobre la que el GPS busca sus rutas;
- los datos que el mapa y las costas usan enteros: la costa, la cobertura del suelo, la orilla, las regiones, los pedraplenes y el agua;
- en la carpeta de arriba, `build.json`, el parte de la construcción: qué provincia salió, con cuántas cosas y con qué versión del formato.

No hace falta tocar nada de eso para jugar. Si abres una versión nueva del juego que cambie el formato de los datos, la ventana lo dice: en vez de «Construyendo Cuba» pone «Actualizando Cuba» y vuelve a construir lo que haga falta, con la misma paciencia y la misma posibilidad de parar y seguir.

## Recuperar el espacio

La isla ocupa unos 2 GB. Si necesitas el disco para otra cosa, borra la carpeta de datos entera y el juego la vuelve a construir la próxima vez que lo abras, con el mismo tiempo de la primera vez. También puedes borrar solo `cuba/` y dejar `runtime/`, que es el juego descomprimido; en ese caso la ventana dirá «Actualizando Cuba» porque el juego ve la marca de una isla que ya no está, y hará el mismo trabajo.

Tus fotos del Álbum y tus diseños del Garaje no corren peligro: no viven en la carpeta de datos sino en la de configuración, junto a las opciones, y borrar la isla no los toca. El capítulo [Opciones y calidad gráfica](#opciones-y-calidad) dice dónde está esa carpeta.
