# Calculating ACE on the sprinkler example
import numpy as np

# Conditional probability tables
clouds_cpt = np.array([0.5, 0.5]) # c, !c
rain_cpt = np.array([
    [0.8, 0.2], # c -> r, c -> !r
    [0.1, 0.9] # !c -> r, !c -> !r
])
sprinkler_cpt = np.array([
    [0.1, 0.9], # c -> s, c -> !s
    [0.5, 0.5] # !c -> s, !c -> !s
])
wet_cpt = np.array([
    [ # r
        [0.95, 0.05], # s -> w, s -> !w
        [0.9, 0.1] # !s -> w, !s -> !w
    ],
    [ # !r
        [0.9, 0.1], # s -> w, s -> !w
        [0.1, 0.9] # !s -> w, !s -> !w
    ]
])

# Indexes
cloudy, sunny = 0, 1
rain, dry = 0, 1
sprinkler_on, sprinkler_off = 0, 1
wet_grass, dry_grass = 0, 1

# ACE = P(Y = 1|do(X = 1)) - P(Y = 1|do(X = 0))
# We want to compute ACE(sprinkler -> wetness)
# ACE_sw = P(w|do(s)) - P(w|do(!s))

# P(w|do(s)) = P(w|s,r)P(r) + P(w|s,!r) = 
# = P(w|s,r)P(r|c)P(c) + P(w|s,r)P(r|!c)P(!c) + P(w|s,!r)P(!r|c)P(c) + P(w|s,!r)P(!r|!c)P(!c) 

P_w_do_s = \
    wet_cpt[sprinkler_on, rain, wet_grass] * rain_cpt[cloudy, rain] * clouds_cpt[cloudy] +\
    wet_cpt[sprinkler_on, rain, wet_grass] * rain_cpt[sunny, rain] * clouds_cpt[sunny] +\
    wet_cpt[sprinkler_on, dry, wet_grass] * rain_cpt[cloudy, dry] * clouds_cpt[cloudy] +\
    wet_cpt[sprinkler_on, dry, wet_grass] * rain_cpt[sunny, dry] * clouds_cpt[sunny]    

# P(w|do(!s)) = P(w|!s,r)P(r) + P(w|!s,!r) = 
# = P(w|!s,r)P(r|c)P(c) + P(w|!s,r)P(r|!c)P(!c) + P(w|!s,!r)P(!r|c)P(c) + P(w|!s,!r)P(!r|!c)P(!c) 

P_w_do_nots =\
    wet_cpt[sprinkler_off, rain, wet_grass] * rain_cpt[cloudy, rain] * clouds_cpt[cloudy] +\
    wet_cpt[sprinkler_off, rain, wet_grass] * rain_cpt[sunny, rain] * clouds_cpt[sunny] +\
    wet_cpt[sprinkler_off, dry, wet_grass] * rain_cpt[cloudy, dry] * clouds_cpt[cloudy] +\
    wet_cpt[sprinkler_off, dry, wet_grass] * rain_cpt[sunny, dry] * clouds_cpt[sunny]    

ACE = P_w_do_s - P_w_do_nots
print(ACE)