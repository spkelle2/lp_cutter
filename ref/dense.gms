set i /1*100/;
alias(i,j, k);
set r /1*1/;

set l /1*21/;
alias (l,m);


positive variable X(i,j);
free variable f;

X.up(i,j) = 1.0;

parameters B(i,j), p, q, n, opt, al(l,l), be(l,l), rate(l,l), gap(l,l), success;
n = card(i);

parameter sol(i,j), delta;
sol(i,j) = 0;
sol(i,j)$(ord(i) < ord(j) and  ord(i) <= n/2 and ord(j) > n/2) = 1;

equations tri1(i,j,k), tri2(i,j,k), tri3(i,j,k), tri4(i,j,k), obj, cardinality;


*tri1(i,j,k)$((B(i,j) = 1 or B(i,k) = 1 or B(j,k) = 1) and ord(i) < ord(j) and ord(j) < ord(k)) .. X(i,j)+X(i,k)  =g= X(j,k);
*tri2(i,j,k)$((B(i,j) = 1 or B(i,k) = 1 or B(j,k) = 1) and ord(i) < ord(j) and ord(j) < ord(k)) .. X(i,j)+X(j,k)  =g= X(i,k);
*tri3(i,j,k)$((B(i,j) = 1 or B(i,k) = 1 or B(j,k) = 1) and ord(i) < ord(j) and ord(j) < ord(k)) .. X(i,k)+X(j,k)  =g= X(i,j);


*tri4(i,j,k)$((B(i,j) = 1 or B(i,k) = 1 or B(j,k) = 1) and  ord(i) < ord(j) and ord(j) < ord(k)) .. X(i,j)+X(j,k)+X(i,k)  =l= 2;

tri1(i,j,k)$(ord(i) < ord(j) and ord(j) < ord(k)) .. X(i,j)+X(i,k)  =g= X(j,k);
tri2(i,j,k)$(ord(i) < ord(j) and ord(j) < ord(k)) .. X(i,j)+X(j,k)  =g= X(i,k);
tri3(i,j,k)$(ord(i) < ord(j) and ord(j) < ord(k)) .. X(i,k)+X(j,k)  =g= X(i,j);


tri4(i,j,k)$(ord(i) < ord(j) and ord(j) < ord(k)) .. X(i,j)+X(j,k)+X(i,k)  =l= 2;


cardinality .. sum((i,j)$(ord(i)<ord(j)), X(i,j)) =e= n*n/4;

obj .. f =e= sum((i,j)$(ord(i)<ord(j)), B(i,j)*X(i,j));

model test /all/;

option lp = cplex;
option solPrint = off;
option limcol = 0;
option limrow = 0;

test.optfile = 1;

$onecho> cplex.opt
lpmethod 4
barepcomp 1e-4
$offecho

execseed = 1 + gmillisec(jnow);


loop(l,

   p = 0.0 + (ord(l)-1)/(card(l)-1)*1.0;
   loop(m,

   q = 0.0 + (ord(m)-1)/(card(m)-1)*1.0;

*stochastic block model


 if (p > q,


   success = 0;
   loop(r,


B(i,j)$(ord(i) < ord(j)) =0;
loop((i,j)$(ord(i)<ord(j)),

     if(ord(i) <= n/2,

        if (ord(j) <= n/2,
*       first block
           if (uniform(0.0, 1.0) < p,
              B(i,j) = 1;
           );
        else
*       cross blocks
           if (uniform(0.0, 1.0) < q,
              B(i,j) = 1;
              B(j,i) = 1;
           );
        );

     else
         if(ord(j) > n/2,
*    second block
           if (uniform(0.0, 1.0) < p,
              B(i,j) = 1;

           );
         );
      );
);

solve test using lp minimizing f;

delta = 0;
loop(i,
   loop(j$(ord(i) < ord(j)),
      delta = delta + (X.l(i,j)-sol(i,j))*(X.l(i,j)-sol(i,j));
   );
);

       if (sqrt(delta/(n*n)) < 1e-4,
          success = success + 1;
          );

*opt =sum((i,j)$(ord(i) < ord(j) and ord(i) <= n/2 and ord(j) > n/2),B(i,j));

*       if (abs(f.l-opt) < 1e-3,
*          success = success + 1;
*          );


);



);
   rate(l,m) = success/card(r);
   al(l,m) = p;
   be(l,m) = q;
   gap(l,m) = p - q;

);
);


display al, be, rate;

