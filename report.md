# Activity report

## Are there different behaviors?
Nonostante possa sembrare interessante, il campo `gait→footprints` non è stato
utilizzato in quanto è un dato "puntuale" che non fornisce informazioni su tutto
il dominio temporale. Un VSR che stia *saltando* potrebbe essere
classificato erroneamente come *strisciante* per un istante di tempo in cui
esaminando il corrispondente `gait→footprints` si deduca che il robot è molto
aderente al terreno, mentre in tale istante il robot potrebbe essere solamente
"atterrato" temporaneamente.

Il problema sarebbe risolto se avessimo a disposizione l'intera evoluzione
temporale di `gait→footprints`, ma abbiamo solamente il valore per il miglior
robot della popolazione in un dato istante di tempo.

Il campo `center.spectrum` invece contiene informazioni su tutto il dominio
temporale.

File: `2dhmsr/tasks/locomotion/Outcome.java`,
`2dhmsr/behavior/BehaviorUtils.java`
