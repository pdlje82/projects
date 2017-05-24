from random import randint

count = {'a': .120, 'b': .120, 'c': .100, 'i': 1.4074407676308254}

maxQ = 0.0
for action in count:
    action_value = count[action]
    if action_value > maxQ:
        maxQ = action_value


actionlist_maxQ = [k for k, v in count.items() if v >= ( maxQ - 0.05 * maxQ ) and v <= ( maxQ + 0.05 * maxQ )]
action = actionlist_maxQ[randint(0, len(actionlist_maxQ)-1)]

print actionlist_maxQ
print action
