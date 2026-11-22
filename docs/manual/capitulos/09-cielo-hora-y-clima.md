# El cielo, la hora y el clima {#cielo-hora-y-clima}

El juego lleva un reloj, y ese reloj manda el sol, la luna, las estrellas y el tiempo que hace. La hora que eliges en el modo libre no es un adorno: a las 19:30 del verano el sol está bajo y la luz es naranja, y a las 03:00 la calle está oscura y el cielo lleno de estrellas. Este capítulo cuenta cómo se mueve todo eso y qué puedes esperar de cada hora y de cada mes.

## El reloj

El paseo arranca a las 10:00 del día de hoy, y el reloj corre a la velocidad de verdad. En el modo libre puedes cambiar la hora, que va de media en media, el día de la semana y la velocidad del reloj: ×1, ×10 o ×60. A ×60, una hora del juego pasa en un minuto tuyo, y puedes ver caer la tarde mientras manejas. La pausa detiene el reloj, y con él el tráfico, el clima y el cielo.

La hora es la misma en toda la isla, y el sol, la luna, las estrellas y el clima se calculan para la posición de La Habana. Por eso se ven iguales en cualquier provincia: el juego usa un solo reloj y un solo cielo.

Si quieres una fecha completa, no solo la hora y el día, abre el juego con ella:

```bash
contravia play --time "2026-07-18 23:00"
```

También valen una hora sola, como `contravia play --time "17:30"`, o un día y una hora, como `contravia play --time "Sun 08:00"`. Sin esa orden, el reloj arranca a las 10:00 de hoy.

## El sol y las sombras

El sol no está inventado: sigue las ecuaciones de posición solar de la NOAA, con La Habana en 23,1 grados de latitud norte y 82,4 de longitud oeste. Eso quiere decir que sale y se pone a las horas que le tocan según la época del año.

| Fecha | Sale | Se pone |
|---|---|---|
| 21 de junio | 6:49 | 20:15 |
| 21 de diciembre | 7:11 | 17:46 |

En junio, al mediodía, el sol cae casi a plomo sobre La Habana. En diciembre se queda bajo, a 43 grados sobre el horizonte, y las sombras son largas todo el día.

La luz cambia con la altura del sol. Cuando está bajo, el color se va al naranja y las sombras se estiran; cuando sube, la luz se vuelve blanca. Al amanecer y al atardecer el cielo se tiñe de naranja y violeta por el lado del sol. Las sombras se dibujan solo si el sol pasa de 3 grados, si hay menos de 85 % de nubes y si no llueve fuerte; con el sol bajo o el cielo cargado, el juego apaga las sombras y todo queda más plano.

<!-- shot: contravia play --place L0333 --time "2026-06-21 06:50" --shot | El sol saliendo sobre el Capitolio -->

## La luna, los planetas y las estrellas

La luna también está calculada de verdad, con sus principales términos astronómicos. Muestra su fase: el lado iluminado se ve claro y el oscuro se dibuja del color del cielo, así que la forma delgada de la luna nueva se adivina apenas. De noche ilumina lo justo para ver los bordes del camino, y de día se ve cuando está sobre el horizonte.

Los planetas son cinco: Mercurio, Venus, Marte, Júpiter y Saturno. Cada uno lleva su brillo y su color, se mueven despacio entre las estrellas y se ven como puntos, no como discos. Venus es el más brillante.

El cielo de noche trae 8 404 estrellas del Yale Bright Star Catalogue y 80 000 puntos de la Vía Láctea. No todas se ven a la vez: la luz de la ciudad, la luna, el crepúsculo y las nubes le van quitando las más débiles. En una noche despejada y sin luna, con la luz de la ciudad como viene puesta, el juego dibuja las estrellas hasta la magnitud 4,5. La Vía Láctea necesita un cielo más oscuro que ese, así que con la luz de la ciudad no aparece.

<!-- shot: contravia play --place M004 --time "2026-07-18 23:00" --shot | El cielo de noche sobre el Malecón -->

De noche el horizonte queda con un resplandor naranja, la luz que la ciudad devuelve al cielo. Esa luz se puede apagar con un archivo de ajustes del cielo: `contravia play --sky cielo.json`, con `{"city_glow": 0}` dentro. En una noche sin luna, el cielo se pone negro de verdad y salen todas las estrellas del catálogo.

## El clima de La Habana

El tiempo que hace viene de una cadena aprendida de treinta años de datos de ERA5, hora por hora y mes por mes. Tiene cinco estados: Sol, Algunas nubes, Nublado, Lluvia y Tormenta, los cinco que fija la fila «Clima» del modo libre además de «Como hoy».

El clima de un día se sortea una sola vez, con números sembrados por la fecha. Eso quiere decir que el 14 de octubre de 2026 hace siempre el mismo tiempo, y que dos partidas distintas se encuentran la misma tarde. Para que un día no empiece de golpe, el juego arrastra el clima de los tres días anteriores: lo que pasó anoche decide con qué cielo amanece hoy.

De febrero a abril el cielo está despejado casi la mitad de las horas y apenas llueve, entre el 1 % y el 2 % de las horas. De junio a octubre hay muchas más nubes; en junio, más de la mitad de las horas están nubladas. La lluvia ocupa poco: entre el 1 % y el 5 % de las horas, según el mes. Las tormentas son raras, alguna tarde de junio a octubre, casi siempre entre las dos y las cinco. Si quieres ver una, elige «Tormenta» en la fila «Clima».

Dentro de una hora el cielo no cambia de golpe: las nubes y la lluvia pasan de un estado al siguiente en los primeros diez minutos. Y cuando escampa, el asfalto sigue mojado cerca de una hora.

## La lluvia y la tormenta

La lluvia se dibuja alrededor del carro, en una caja de 60 metros que se mueve contigo. En la calidad que viene puesta son hasta 3 000 rayas de agua, que caen a 9 metros por segundo y el viento empuja de lado. Cuanta más lluvia, más rayas se ven.

<!-- shot: contravia play --place M004 --time "2026-10-14 15:00" --sky cielo-tormenta.json --shot | La lluvia sobre el Malecón -->

Con la lluvia se cierra la niebla, que en una tormenta llega a unos 200 metros, y el cielo se pone gris. Se oye el ruido de la lluvia alrededor del carro y las gomas levantan agua. En tormenta caen truenos: uno cada 25 segundos de promedio, entre 300 metros y 6 kilómetros de distancia. Como el sonido tarda en llegar, el trueno se oye hasta 17 segundos y medio después del rayo, y suena más seco y corto si cayó cerca, más largo y grave si cayó lejos.

Si las gotas te molestan en la pantalla, «Lluvia en pantalla» en Opciones las quita. El clima sigue igual, el sonido sigue igual y la calle sigue mojada; lo que desaparece es el agua dibujada delante de la cámara.

## Lo que la lluvia cambia y lo que no

La lluvia cambia lo que ves y lo que oyes. El asfalto se pone oscuro y brillante, el cielo se carga, la niebla se cierra y el ruido del agua tapa parte del sonido de la calle. El chirrido de una goma pasada de vueltas también se oye más apagado cuando está mojada.

Lo que no cambia es el agarre. Hoy el carro agarra igual en seco que en mojado: la lluvia no lo hace derrapar ni alarga la frenada. Es una de las cosas que el juego todavía no simula, y conviene saberlo para no esperar un comportamiento que no está.

## Un día entero

Para ver cómo cambia la luz sin manejar, pon el reloj a ×60 en el modo libre y déjalo correr desde la mañana. La tarde se va, el sol se pone por el oeste, el cielo pasa por el naranja y el violeta, y a los pocos minutos de juego están las estrellas. Si abres con una fecha de invierno, la tarde dura menos y anochece antes de las seis.

También puedes pedirle al juego que grabe un día entero o que dibuje un año de clima, sin abrir la ventana:

```bash
contravia timelapse tarde.mp4
contravia skymap ano.png
```

El primero arma un video de un día sobre el Capitolio, desde las 4:30 de la madrugada. El segundo dibuja un año de la cadena del clima, con los días en vertical y las horas en horizontal, para que se vean de un golpe las tardes de lluvia y los meses secos.
