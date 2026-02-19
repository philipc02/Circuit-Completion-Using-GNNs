* SPICE Netlist for the given circuit

   .model NMOS NMOS level=1

   V1 6 0 DC Vin
   RD 7 3 RD_VALUE
   RS 6 4 RS_VALUE
   CGD 2 7 CGD_VALUE
   CGS 2 4 CGS_VALUE
   CDB 7 3 CDB_VALUE

   M1 7 2 5 5 NMOS

   .end