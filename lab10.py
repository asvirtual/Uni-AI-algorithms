import numpy as np
# Setup arrays with: symbolic names for outcomes (not currently used), utilities of outcomes, and
# probabililites of those outcomes
office_outcomes = ["work", "distracted", "colleague"]
print('office_outcomes = ', office_outcomes)
u_office_outcomes = np.array([8, 1, 5])
print('U(office_outcomes) = ', u_office_outcomes)
p_office_outcomes_office = np.array([0.5, 0.3, 0.2])
print('P(office_outcomes|office) =', p_office_outcomes_office)

# The weighted utility ofeach outcome is each to compute by pairwise multiplication
eu_office_outcomes = u_office_outcomes * p_office_outcomes_office
print('EU by outcome =', eu_office_outcomes)
# Summing the weighted utilities gets us the expected utility
eu_office = np.sum(eu_office_outcomes)
eu_office_2 = np.dot(u_office_outcomes, p_office_outcomes_office) # alternatives
print(eu_office_2)
print('EU(office) = ', eu_office)


# The coffee calculation is the same as the office calculation, first set up arrays
coffee_outcomes = ["caffeination", "spillage"]
print('coffee_outcomes = ', coffee_outcomes)
u_coffee_outcomes = np.array([10, -20])
print('U(coffee_outcomes) = ', u_coffee_outcomes)
p_coffee_outcomes_coffee = np.array([0.95, 0.05])
print('P(coffee_outcomes|coffee) =', p_coffee_outcomes_coffee)
print('\n')
# Then compute the expected utility
eu_coffee_outcomes = u_coffee_outcomes * p_coffee_outcomes_coffee
print('EU by outcome =', eu_coffee_outcomes)
eu_coffee = np.sum(eu_coffee_outcomes)
print('EU(coffee) = ', eu_coffee)

# MEU CRITERION
if eu_office > eu_coffee:
    print('Office is the MEU choice')
else:
    print('Coffee is the MEU choice')

# MAXIMAX CRITERION
# The utility of each choice is the max utility of their outcomes
max_u_office = np.max(u_office_outcomes)
print('MaxU(office) =', max_u_office)
max_u_coffee = np.max(u_coffee_outcomes)
print('MaxU(coffee) =', max_u_coffee)
print('\n')
# The decision criterion is then to pick the outcome with the highest utility:
if max_u_office > max_u_coffee:
    print('Office is the Maximax choice')
else:
    print('Coffee is the Maximax choice')

# MAXIMIN CRITERION
# The utility of each choice is the max utility of their outcomes
min_u_office = np.min(u_office_outcomes)
print('MinU(office) =', min_u_office)
min_u_coffee = np.min(u_coffee_outcomes)
print('MinU(coffee) =', min_u_coffee)
print('\n')
# The decision criterion is then to pick the outcome with the highest utility:
if min_u_office > min_u_coffee:
    print('Office is the Maximin choice')
else:
    print('Coffee is the Maximin choice')

