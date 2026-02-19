spice
* SPICE Netlist for provided schematic

* Transistor Q1 and Q2 are NPN BJTs
Q1 Collector1 Base1 Emitter1 Q1
Q2 Collector2 Base2 Emitter2 Q2

* Capacitors
C1 Collector1 VCC 0.1uF
C2 Emitter2 3 0.1uF

* Current Source
I1 3 2 DC 1mA

* Connections
VCC 1 0 DC 15V
VEE 2 0 DC -15V

* Input and Output
Vi Base1 0 DC 1V
Vbias Base2 0 DC 2.5V
Vo Emitter2 0

* Node Assignments
* 1 - Connected to VCC
* 2 - Connected to VEE
* 3 - Current Source Negative Terminal
* 4 - Vbias