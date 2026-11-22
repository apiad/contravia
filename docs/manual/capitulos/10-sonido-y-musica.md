# Sonido y música {#sonido-y-musica}

El juego no trae ni un solo archivo de audio grabado. Todo lo que oyes se fabrica mientras juegas, nota por nota y golpe por golpe, con las mismas reglas para todos: el motor sale de las revoluciones que lleva, la lluvia del agua que está cayendo y la música del lugar donde estás. Por eso el sonido cambia cuando cambias de barrio o de hora.

## De dónde sale el sonido

El sonido se mezcla en cinco canales: motor, efectos, ambiente, música e interfaz. Cuatro de ellos tienen su fila de volumen en Opciones, y el general manda sobre todos. El motor y los efectos se oyen solo manejando; en los menús y en la pausa quedan en silencio, y el ambiente y la música siguen.

## El carro

El motor se arma con las revoluciones y con lo que pisas. Cada familia tiene su voz: el almendrón de ocho cilindros burbujea y petardea cuando levantas el pie a altas revoluciones, el Ruso y el Moscovita suenan más finos, la patrulla y los carros modernos son más lisos, y las guaguas y los camiones tienen el golpeteo del diésel. La metroguagua añade un silbido de turbo cuando acelera.

Las gomas cantan según el suelo. Sobre el asfalto el ruido es grave y bajo; fuera de la calle se vuelve más brillante y más fuerte. Si pasas una goma de su límite de agarre, chirría, y el chirrido se oye más apagado con el piso mojado. El viento sube de tono y de volumen con la velocidad.

La bocina suena mientras tengas H apretada, y no todas son iguales: el almendrón tiene dos tonos, el Ruso y el Moscovita uno, las guaguas y los camiones una bocina de aire y las motos un pitido delgado. Un golpe fuerte suena a metal y, si es muy fuerte, a vidrio roto; un bordillo o un hueco suenan a un golpe seco, más fuerte cuanto más se hundió la suspensión.

## La calle

Del tráfico se oyen los cuatro carros que más suenan en cada momento, escogidos por lo cerca que están, lo rápido que van y lo pesados que son. El resto de los carros a menos de 80 metros es un rumor de fondo que crece con la cantidad. Una guagua que frena suelta el aire de los frenos, y un carro que lleva un rato esperando detrás de ti, parado, te toca la bocina.

## El tiempo y el agua

La lluvia tiene su propio ruido: un siseo constante, gotas que repiquetean y el agua que levantan las gomas. En tormenta se oyen truenos, cada uno con su distancia: los de cerca son secos y cortos, los de lejos llegan tarde y ruedan largo.

Si el carro se mete en el mar o en un río hondo, se oye el golpe del agua, después todo se apaga como si te taparan los oídos y empiezan a subir burbujas mientras el carro se hunde. Cuando vuelve a la orilla, el sonido regresa de golpe.

## El lugar

El ambiente y la música cambian con el sitio. El juego mira lo construido que está el lugar, lo cerca que está el mar y la hora, y escoge uno de cinco ambientes:

| Ambiente | Dónde | Qué se oye | Música |
|---|---|---|---|
| Ciudad | El centro denso | Un rumor grave de tráfico y, de vez en cuando, un perro | Son montuno |
| Barrio | Los barrios fuera del centro denso | Un rumor más suave, perros y un gallo al amanecer | Rumba guaguancó |
| Campo | Fuera de las ciudades | Pájaros, un gallo, una vaca a lo lejos y algún perro | Guajira |
| Costa | A menos de 300 metros del mar | Olas y gaviotas | Changüí |
| Noche | Cuando ya oscureció, en cualquier lugar | Grillos, ranas y, de vez en cuando, un tres desde una ventana | Bolero |

El cambio no es de golpe: el juego espera unos diez segundos en el ambiente nuevo y después cruza un ambiente con el otro en tres segundos. Así, cruzar un parque no te cambia la música a mitad de la calle.

Un lugar puede pedir su propio ambiente o su propia música. Guantánamo, por ejemplo, suena siempre a changüí, aunque estés lejos del mar.

## La música

Hay cinco estilos y cada uno se toca con sus instrumentos. El son lleva tres, bajo, claves, bongó y maracas. El guaguancó lleva tres tumbadoras, clave, palitos y cajón. La guajira lleva laúd, guitarra, güiro, claves y bajo. El changüí lleva tres, marímbula, bongó, güiro y maracas. El bolero lleva guitarra de nylon, bajo, bongó, maracas y flauta.

La música no está grabada: se compone compás a compás mientras manejas. El mismo día y el mismo estilo dan siempre la misma música, así que puedes volver a una fecha y reconocer lo que sonaba. El juego va componiendo dos compases por delante para que no haya silencios, y cuando cambia de estilo cruza uno con otro en tres segundos.

En los menús suena un son más suave, con menos percusión. En la pausa la música baja 10 decibelios, para que puedas dejarla de fondo.

## Los volúmenes

En Opciones hay cinco filas de volumen, de 0 a 10:

| Fila | Qué controla | Cómo viene |
|---|---|---|
| Volumen general | Todo el sonido del juego | 8 |
| Motor | El motor de tu carro | 7 |
| Efectos | Gomas, viento, choques, bocina, lluvia y tráfico | 7 |
| Ambiente | La ciudad, el campo y la costa alrededor | 6 |
| Música | La música de fondo | 5 |

La interfaz no tiene fila propia: sigue al volumen general. Los menús, el disparo de la cámara, la ficha de un lugar nuevo y el aviso de un nivel de ayuda suenan por ahí.

Si no quieres oír nada, abre el juego con `contravia play --mute`: no suena nada, ni siquiera en los menús. Una captura de pantalla tampoco suena nunca. Y si quieres guardar el sonido de un paseo, `contravia play --audio-out paseo.wav` graba la mezcla en ese archivo y deja al lado un WAV por cada canal.

## Nada es una grabación

No hay música con licencia de otro, ni ruidos de biblioteca, ni voces. Cada sonido sale de una receta que el juego ejecuta en el momento, y la música de cada estilo está escrita como una regla que arma los compases. Eso es lo que permite que la isla suene distinta en cada esquina y que el juego pueda cambiar un instrumento o un ambiente sin traer un solo archivo nuevo.
