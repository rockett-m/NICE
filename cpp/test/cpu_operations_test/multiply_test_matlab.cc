// The MIT License (MIT)
//
// Copyright (c) 2016 Northeastern University
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "include/cpu_operations.h"
#include "include/matrix.h"

///////////
a = [ 5     9     1     4     2     9     3     6     8     0
      5     5     3     8     4     9     7     8     6     7
      3     6     0     8     5     1     9     0     3     7
      8     6     6     3     9     7     0     3     3     5
      2     7     9     4     7     1     2     2     6     0
      9     9     7     6     9     1     5     4     7     0
      3     7     7     5     0     3     8     3     7     9
      6     8     1     1     6     3     7     9     9     2
      2     9     3     0     9     3     5     1     0     8
      0     7     5     1     2     2     0     2     4     3 ];
///////////

5,5,3,8,2,9,3,6,2,0,9,5,6,6,7,9,7,8,9,7,1,3,0,6,9,7,7,1,3,5,4,8,8,3,4,6,5,1,0,1,2,4,5,9,7,9,0,6,9,2,9,9,1,7,1,1,3,3,3,2,3,7,9,0,2,5,8,7,5,0,6,8,0,3,2,4,3,9,1,2,8,6,3,3,6,7,7,9,0,4,0,7,7,5,0,0,9,2,8,3,

scalar = 0;

product = [ 0     0     0     0     0     0     0     0     0     0
            0     0     0     0     0     0     0     0     0     0
            0     0     0     0     0     0     0     0     0     0
            0     0     0     0     0     0     0     0     0     0
            0     0     0     0     0     0     0     0     0     0
            0     0     0     0     0     0     0     0     0     0
            0     0     0     0     0     0     0     0     0     0
            0     0     0     0     0     0     0     0     0     0
            0     0     0     0     0     0     0     0     0     0
            0     0     0     0     0     0     0     0     0     0 ];

EXPECT_EQ(a * scalar, product); //rename 
///////////////

//////////////
scalar1 = -1;

a_scalar1 = [  -5    -9    -1    -4    -2    -9    -3    -6    -8     0
               -5    -5    -3    -8    -4    -9    -7    -8    -6    -7
               -3    -6     0    -8    -5    -1    -9     0    -3    -7
               -8    -6    -6    -3    -9    -7     0    -3    -3    -5
               -2    -7    -9    -4    -7    -1    -2    -2    -6     0
               -9    -9    -7    -6    -9    -1    -5    -4    -7     0
               -3    -7    -7    -5     0    -3    -8    -3    -7    -9
               -6    -8    -1    -1    -6    -3    -7    -9    -9    -2
               -2    -9    -3     0    -9    -3    -5    -1     0    -8
                0    -7    -5    -1    -2    -2     0    -2    -4    -3 ];

EXPECT_EQ(a * scalar1, a_scalar1);
////////////

////////////
scalar_2 = 0.25;

a_scalar2 = [  1.2500    2.2500    0.2500    1.0000    0.5000    2.2500    0.7500    1.5000    2.0000         0
               1.2500    1.2500    0.7500    2.0000    1.0000    2.2500    1.7500    2.0000    1.5000    1.7500
               0.7500    1.5000         0    2.0000    1.2500    0.2500    2.2500         0    0.7500    1.7500
               2.0000    1.5000    1.5000    0.7500    2.2500    1.7500         0    0.7500    0.7500    1.2500
               0.5000    1.7500    2.2500    1.0000    1.7500    0.2500    0.5000    0.5000    1.5000         0
               2.2500    2.2500    1.7500    1.5000    2.2500    0.2500    1.2500    1.0000    1.7500         0
               0.7500    1.7500    1.7500    1.2500         0    0.7500    2.0000    0.7500    1.7500    2.2500
               1.5000    2.0000    0.2500    0.2500    1.5000    0.7500    1.7500    2.2500    2.2500    0.5000
               0.5000    2.2500    0.7500         0    2.2500    0.7500    1.2500    0.2500         0    2.0000
                    0    1.7500    1.2500    0.2500    0.5000    0.5000         0    0.5000    1.0000    0.7500 ];

EXPECT_EQ(a*scalar2, a_scalar2);
///////////



///////////
b = [ 4     0     7     8     3     3     7     2     8     8
      1     0     7     9     1     4     5     2     3     5
      7     4     8     4     8     4     5     0     2     1
      4     7     7     7     5     0     2     7     1     5
      2     8     9     9     9     5     5     8     6     7
      3     6     5     8     4     8     9     7     5     4
      4     3     9     9     2     5     4     7     0     2
      4     8     9     4     6     2     6     1     4     2
      6     2     5     8     9     8     4     5     4     9
      6     0     8     9     6     7     7     8     7     4 ];
///////////

///////////
product_ab = [ 167   175   318   358   220   228   264   202   186   246
               251   251   452   467   314   287   341   314   243   284
               159   135   321   355   191   184   198   255   146   206
               191   189   365   381   277   238   295   228   245   259
               163   160   297   288   242   175   190   155   138   195
               217   209   426   429   304   233   281   231   222   302
               237   143   382   395   260   254   269   242   182   224
               194   188   387   398   266   247   278   223   212   269
               137   125   315   335   196   206   228   215   178   183
               106    79   186   191   144   131   139    97    99   119 ];
EXPECT_EQ(a * b, product_ab);
//////////

//////////
c = [  1     0     0     0     0     0     0     0     0     0
       0     1     0     0     0     0     0     0     0     0
       0     0     1     0     0     0     0     0     0     0
       0     0     0     1     0     0     0     0     0     0
       0     0     0     0     1     0     0     0     0     0
       0     0     0     0     0     1     0     0     0     0       
       0     0     0     0     0     0     1     0     0     0
       0     0     0     0     0     0     0     1     0     0
       0     0     0     0     0     0     0     0     1     0
       0     0     0     0     0     0     0     0     0     1 ];
////////

////////
product_ac = [ 5     9     1     4     2     9     3     6     8     0
               5     5     3     8     4     9     7     8     6     7
               3     6     0     8     5     1     9     0     3     7
               8     6     6     3     9     7     0     3     3     5
               2     7     9     4     7     1     2     2     6     0
               9     9     7     6     9     1     5     4     7     0
               3     7     7     5     0     3     8     3     7     9
               6     8     1     1     6     3     7     9     9     2
               2     9     3     0     9     3     5     1     0     8
               0     7     5     1     2     2     0     2     4     3 ];
EXPECT_EQ(a * c, product_ac);
/////////

/////////
d = [ 5     5     3     8     2     9     3     6     2     0
      9     5     6     6     7     9     7     8     9     7
      1     3     0     6     9     7     7     1     3     5
      4     8     8     3     4     6     5     1     0     1
      2     4     5     9     7     9     0     6     9     2
      9     9     1     7     1     1     3     3     3     2
      3     7     9     0     2     5     8     7     5     0
      6     8     0     3     2     4     3     9     1     2
      8     6     3     3     6     7     7     9     0     4
      0     7     7     5     0     0     9     2     8     3 ];
/* Equals a.transpose() */

product_db = [ 157   199   304   310   198   172   238   192   158   194
               304   279   532   548   394   350   407   347   309   357
               150   191   313   350   222   212   223   276   163   207
               148   144   297   299   181   178   221   163   140   163
               214   259   368   384   323   247   279   261   208   272
               139   106   282   304   155   139   197   155   157   211
               191   172   345   321   229   217   250   159   148   177
               126   144   286   280   156   147   218   144   161   174
               192   216   408   396   243   227   310   236   226   230
               186   122   303   306   208   190   181   178   101   173 ];
EXPECT_EQ(d * b, product_db);
////////
