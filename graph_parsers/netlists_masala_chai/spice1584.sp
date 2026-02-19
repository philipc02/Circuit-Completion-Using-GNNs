spice
* Circuit Description
VCC VCC 0 DC 15V

* Transistor Q1
Q1 Vout Vin1 P NPN

* Transistor Q2
Q2 Vout Vin2 P NPN

* Resistors
RC1 VCC Vout 1k
RC2 VCC Vout 1k
REE P 0 1k

* Current Source IEE
IEE P 0 DC 1mA

* Model Definitions
.model NPN NPN (IS=1E-14 BF=100)