plaintext
* NMOS Amplifier Circuit

M1 Vout1 Vin1 Vout1 Vout1 NMOS_MODEL
I1 Vout1 GND DC 0.2mA

* Voltage source for VDD
VDD VDD GND DC <VDD_value>

* Input Voltage source
Vin1 Vin1 GND DC <Vin1_value>

* Models (add this line if you have NMOS model defined)
*.model NMOS_MODEL NMOS (LEVEL=1 TO=0 TP=0)
.end