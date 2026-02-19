spice
* Simple NPN transistor circuit

VCC 3 0 DC 2.5V

R1 3 4 5k
RB 4 0 RB_value
R2 2 0 1k

Q1 3 4 2 NPN

.model NPN NPN (Is=1e-14 Vaf=100 Bf=100)

.end