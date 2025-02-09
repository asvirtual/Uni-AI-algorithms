import numpy as np


class CPT:
    def __init__(self, probabilities):
        self._probabilities, self._ordering = CPT.probabilities_to_arr(probabilities)


    def get(self, keys):
        if isinstance(keys, tuple):
            to_return = self._probabilities
            for key in keys:
                to_return = to_return[int(key)]

            return to_return
        
        if isinstance(keys, dict):
            if len(keys) == 0:
                return self._probabilities

            indexes = []
            for key in self._ordering:
                indexes.append(keys[key])

            return self.get(tuple(indexes))

            
        return self._probabilities[keys]
    

    def probabilities_to_arr(dictionary, index=0):
        if not isinstance(dictionary, dict):
            raise ValueError("[ERROR] Input must be a dictionary")

        arr = []
        ordering = {}
        for key, val in dictionary.items():
            ordering[key] = index
            for item in val:
                if isinstance(item, dict):
                    new_arr, new_ordering = CPT.probabilities_to_arr(item, index + 1)
                    arr.append(new_arr)
                    ordering.update(new_ordering)
                else:
                    arr.append(item)

        return np.array(arr), ordering
    

    def __getitem__(self, keys):
        return self.get(keys)
    

# cpt = CPT({
#     "S": [
#         {
#             "R": [
#                 {
#                     "A": [
#                         [
#                            0.95,
#                            0.05
#                         ],
#                         [
#                            0.90,
#                            0.10
#                         ]
#                     ]
#                 },
#                 {
#                     "A": [
#                         [
#                            0.80,
#                            0.20
#                         ],
#                         [
#                            0.40,
#                            0.60
#                         ]
#                     ]
#                 }
#             ],
#         },
#         {
#             "R": [
#                 {
#                     "A": [
#                         [
#                            0.95,
#                            0.05
#                         ],
#                         [
#                            0.90,
#                            0.10
#                         ]
#                     ]
#                 },
#                 {
#                     "A": [
#                         [
#                            0.80,
#                            0.20
#                         ],
#                         [
#                            0.40,
#                            0.60
#                         ]
#                     ]
#                 }
#             ],
#         }
#     ]
# })

# print(cpt[{ "R": 1, "A": 1, "S": 0 }])

cpt = CPT({
    "C": [
        [0.8, 0.2],
        [0.1, 0.9]
    ]
})

# print(cpt[{ "C": 0 }])
    

'''
    Example of input data for a CPT object:

    {
        "S": [
            {
                "R": [
                    {
                        "A": [
                            0.95,
                            0.90
                        ]
                    },
                    {
                        "A": [
                            0.90,
                            0.10
                        ]
                    }
                ],
            },
            {
                "R": [
                    {
                        "A": [
                            0.95,
                            0.90
                        ]
                    },
                    {
                        "A": [
                            0.90,
                            0.10
                        ]
                    }
                ]
            }
        ]
    }
'''