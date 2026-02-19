plaintext
* SPICE Netlist for NPN Transistor Circuit

V1 Vin 0 DC Vin
V2 2 0 DC 5V

RB Vin B 200k
RC 2 C 4k
RE E 0 Re

Q1 C B E NPN

* Node mapping:
* B -> Base
* C -> Collector
* E -> Emitter

.model NPN NPN
.end