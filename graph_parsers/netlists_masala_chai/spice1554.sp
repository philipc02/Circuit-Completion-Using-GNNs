plaintext
* Differential Amplifier Circuit

Q1 N Vin1 IEE NPN
Q2 Vout Vin2 IEE NPN
Q3 VCC N VCC PNP
Q4 VCC Vout VCC PNP
I1 IEE 0 DC IEE_VALUE
V1 VCC 0 DC VCC_VALUE

* Define the model parameters for NPN and PNP transistors
.model NPN NPN (IS=1E-14 BF=100)
.model PNP PNP (IS=1E-14 BF=100)

* Assign values to sources
.param IEE_VALUE=1mA
.param VCC_VALUE=15V

.end