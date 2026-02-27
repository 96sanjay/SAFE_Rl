==========================================================================
  C3 VIOLATION BREAKDOWN - FULL YEAR, RBC BASELINE (NO PSF)
===========================================================================
  Total (building x step) slots: 148903
  Total C3 violations:           20054 (13.47%)
  No violation:                  128849 (86.53%)

  CATEGORY BREAKDOWN
  A)  Base load alone > 3.47 kW      UNFIXABLE:    7383 (4.96%)
  B1) EV-caused, battery CAN fix     FIXABLE:      7658 (5.14%)
  B2) EV-caused, battery CANNOT fix  UNFIXABLE:    1592 (1.07%)
  C)  RBC battery caused it          FIXABLE:      3421 (2.30%)

  SUMMARY
  Unfixable floor (A + B2):    8975 (6.03%)
  Fixable by PSF  (B1 + C):   11079 (7.44%)
  Theoretical BEST C3 = 6.03%
  Current PSF C3      = 11.61%
  Gap to improve      = 5.58%

  PER-BUILDING DETAIL
  Bldg EVs  MaxkW  Viols  A(base)  B1(fix) B2(nofix)  C(batt)      EVkWh
     0   1   11.0   3321     2197      949       100       75    12119.4
     1   0    0.0    277       90        0         0      187        0.0
     2   0    0.0    155       18        0         0      137        0.0
     3   1   22.0   1136      239      155       599      143    15359.2
     4   1    7.4   2231     1078     1115         0       38     8782.9
     5   0    0.0    183       65        0         0      118        0.0
     6   1   11.0   2361     1309      819        58      175     9998.4
     7   0    0.0    232      104        0         0      128        0.0
     8   0    0.0    171       29        0         0      142        0.0
     9   1    7.4   2550      862     1511         0      177    12813.9
    10   0    0.0    296       48        0         0      248        0.0
    11   1    7.4   1713        0     1281         0      432     9907.3
    12   0    0.0    500      279        0         0      221        0.0
    13   0    0.0    426      147        0         0      279        0.0
    14   2   11.0   3349       37     1828       835      649    31550.5
    15   0    0.0    464      336        0         0      128        0.0
    16   0    0.0    689      545        0         0      144        0.0

  EV POWER DISTRIBUTION (when charging)
  Total EV-charging hours: 38926
  Mean / Median:  2.58 / 0.10 kW
  Min / Max:      0.05 / 22.00 kW
  EV power >   3.5 kW:  9650 (24.8%)
  EV power >   7.0 kW:  8595 (22.1%)
  EV power >  10.0 kW:  3655 (9.4%)
  EV power >  13.5 kW:  1432 (3.7%)
  EV power >  20.0 kW:   504 (1.3%)
(citylearn) christmas@friday:/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork$ 