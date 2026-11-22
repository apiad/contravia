# Datos, créditos y licencias {#datos-y-licencias}

Contravía se arma con mapas y datos que no son del juego, y se comparte con licencias que dicen qué puedes hacer con él. Este anexo cuenta de dónde sale cada cosa, qué permiten las licencias en palabras llanas y qué reglas hay para los mods. No sustituye a los textos legales: los archivos que están al final son los que valen.

## Dónde verlo en el juego

En «Opciones», la fila «Créditos y licencias» abre una pantalla con las atribuciones que piden las licencias de los datos y la licencia de cada parte del juego. ESC vuelve a Opciones. La pantalla resume lo que cuenta este anexo.

<!-- shot: contravia play --screen credits --shot | La pantalla «Créditos y licencias» -->

## De dónde salen los datos

| Qué | De dónde | Licencia |
|---|---|---|
| Calles, edificios, lugares y costa | OpenStreetMap, © los colaboradores de OpenStreetMap | Open Database License (ODbL 1.0) |
| Relieve | Copernicus GLO-30, © DLR e.V. 2010-2014 y © Airbus Defence and Space GmbH 2014-2018, bajo COPERNICUS por la Unión Europea y la ESA | Copernicus DEM: gratis, con la línea de crédito que pide |
| Cobertura del suelo en el mapa | ESA WorldCover 2021, con datos modificados de Copernicus Sentinel (2021) procesados por el consorcio ESA WorldCover | CC BY 4.0 |
| Clima | ERA5 horario 1991-2020 del Copernicus Climate Change Service, vía Open-Meteo, y registros NOAA GHCN-Daily | CC BY 4.0 para ERA5, y registros NOAA GHCN-Daily |
| Cielo nocturno | Yale Bright Star Catalogue y NASA Deep Star Maps 2020 (Scientific Visualization Studio) | Dominio público |

Las calles, las casas, los lugares para fotografiar y la línea de la costa vienen de OpenStreetMap. El relieve, que da las lomas y los valles, viene de Copernicus GLO-30. La cobertura del suelo, que dice qué parte del terreno está construida, es de ESA WorldCover. El clima sale de treinta años de ERA5 y de los registros de estaciones del NOAA. Las estrellas y la Vía Láctea salen de dos catálogos de dominio público.

Las licencias de los datos piden que el juego los nombre donde nombra sus datos, y la pantalla de créditos lo hace.

## Las licencias, en palabras llanas

| Parte | Licencia | Qué puedes hacer |
|---|---|---|
| El código | GPL-3.0-or-later | Leerlo, copiarlo, cambiarlo y repartirlo, y cobrar por él. Si repartes una versión cambiada, tiene que ir con su fuente y bajo la misma licencia |
| El arte, la música, la historia y estos documentos | CC BY-NC-SA 4.0 | Usarlos y compartirlos con crédito, sin ganar dinero con ellos; una obra hecha con ellos queda bajo la misma licencia, salvo lo que permite la Licencia de Mods |
| Los datos del mapa | ODbL 1.0 | Usarlos con crédito a OpenStreetMap; si publicas una base de datos hecha con ellos, tiene que quedar bajo la misma licencia |
| Las tipografías | SIL Open Font License 1.1 | Usarlas y repartirlas con el juego, con su licencia al lado |
| El motor, manifoldx | MIT | Usarlo, cambiarlo y repartirlo, con su aviso de copyright |

El código es software libre. El arte y la música no se pueden usar para ganar dinero, aunque sí compartir, y una obra que salga de ellos tiene que dar el crédito y usar la misma licencia. La Licencia de Mods, más abajo, abre una excepción para los mods. Los datos del mapa son abiertos, como manda OpenStreetMap, y por eso el juego puede construirse y repartirse.

## Los mods

Un mod es una obra que necesita el juego instalado para funcionar y que añade o cambia contenido: carros, misiones, actividades, música, personajes, texturas o zonas. La Licencia de Mods de Contravía es un permiso adicional sobre la licencia de las piezas del juego, y no quita ningún derecho que esa licencia ya te dé.

Puedes vender, regalar o monetizar tu mod, aunque use o cambie piezas del juego, siempre que:

- no repartas el juego base ni lo conviertas en un juego independiente;
- no pongas detrás de un pago nada que el juego base tenga gratis;
- te presentes como «un mod para Contravía», sin el logo y sin parecer oficial;
- des el crédito que pide la licencia original.

Lo que hagas desde cero es tuyo, con la licencia que elijas. Los videos, streams y capturas del juego se pueden monetizar. Si tu mod importa código del juego, ese código queda bajo la GPL como cualquier obra basada en él: hay que publicar su fuente, aunque el mod se pueda vender, y las piezas propias del mod siguen siendo de su autor. Una versión nueva de la licencia de mods nunca le quita permiso a lo que una versión anterior ya permitió.

## El nombre y el logo

Las licencias cubren el código, los datos y el arte, pero no dan ningún derecho sobre el nombre «Contravía» ni sobre su logo, que identifican el juego oficial y la tienda oficial.

- Un fork puede decir que está «basado en Contravía», pero tiene que publicarse con otro nombre y sin el logo.
- Un mod sigue la regla de presentarse como «un mod para Contravía», sin logo y sin parecer oficial.
- Los videos, streams, reseñas y artículos pueden usar el nombre para hablar del juego.

Los vehículos llevan los nombres que se usan en La Habana, no los de ninguna fábrica ni modelo, así que tampoco hay marcas de carros en el juego.

## Los archivos

Todo esto está en el repositorio, y cada archivo es el que manda:

- `LICENSE`: el texto de la GPLv3.
- `LICENSES/`: los textos de la ODbL, la CC BY 4.0, la CC BY-NC-SA 4.0, la OFL, la GPL y la Licencia de Mods.
- `MODS.md`: la Licencia de Mods, en inglés y en español.
- `TRADEMARKS.md`: la regla del nombre y el logo, en inglés y en español.
- `README.md`: el resumen de las licencias y las atribuciones de los datos.
- `REUSE.toml`: la licencia de cada carpeta del repositorio, para que no quede ningún archivo sin dueño.
