plaintext
* Transistor Q1
Q1 collector_base_emitter 2 2 3 NPN

* Transistor Q2
Q2 collector_base_emitter 2 2 3 NPN

* Current Source
I1 2 0 DC 200uA

* Voltage Source
V1 6 0 DC Vs

* Resistors
R1 6 2 10k
R2 2 3 10k
R3 3 0 140
R4 2 0 500

* Voltage for the collector
Vcc 2 0 DC 15V

* Output Voltage Node
Vo 2 4 DC