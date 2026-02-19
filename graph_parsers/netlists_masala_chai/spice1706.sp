plaintext
* SPICE Netlist
VDD 7 0 DC <Vdd_value> 
VCM 6 0 DC <Vcm_value>

RD 3 7 <RD_value>
RDP 4 7 <RD_Delta_value>
RSS 2 8 <RSS_value>

M1 5 6 8 8 NMOS
M2 5 4 8 8 NMOS

IEE 8 0 DC <Iee_value>

CSS 2 0 <CSS_value>

* The drain, gate, source, and body terminals of the MOSFET
* are in the order: drain gate source body

* Node numbers according to the annotated image:
* 3: Vout1
* 4: Vout2
* 7: Connected to VDD
* 8: Common node for current source IEE and the body terminals of M1 and M2

* Replace <RD_value>, <RD_Delta_value>, <RSS_value>, <Iee_value>, and <CSS_value>
* with actual component values.
* Replace <Vdd_value> and <Vcm_value> with supply voltage values.