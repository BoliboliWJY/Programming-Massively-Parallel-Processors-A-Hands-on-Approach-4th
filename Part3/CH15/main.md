#answer for CH15

## 1

### a

|   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 |
| 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 
| 2 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| 3 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| 4 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| 5 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| 6 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| 7 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 |

### b

col_ind: [2, 5, 0, 4, 7, 3, 0, 6, 3, 1, 7, 4, 2, 4, 6]

row_ptr: [0, 2, 5, 6, 8, 9, 11, 12, 15] <- n+1 idx

### c

#### i

8 threads; [1, 2, 3] for idx [0 ], [2, 5], [1, 3, 7]

#### ii

8 threads; 

iterate:[7, 5, 2]

label:[2, 3, 2]

#### iii

15 threads;

label:[2, 3, 8] us if active

![边搜索示例](/Part3/CH15/image.png)

#### iv

[1, 2, 3, 2] threads

label:[1, 2, 3, 2] go through all point

## 2

set the terminal path under CH15

## 3

use the web-Google as the example:

```
Graph loaded: 916428 vertices, 5105039 edges
currLevel=0, currSize=1, nextSize=430, use_singleBlock=1, levelsProcessed=4, overflow=1
currLevel=4, currSize=430, nextSize=1017, use_singleBlock=0, levelsProcessed=0, overflow=0
currLevel=5, currSize=1017, nextSize=2500, use_singleBlock=0, levelsProcessed=0, overflow=0
currLevel=6, currSize=2500, nextSize=6870, use_singleBlock=0, levelsProcessed=0, overflow=0
currLevel=7, currSize=6870, nextSize=22533, use_singleBlock=0, levelsProcessed=0, overflow=0
currLevel=8, currSize=22533, nextSize=58616, use_singleBlock=0, levelsProcessed=0, overflow=0
currLevel=9, currSize=58616, nextSize=113798, use_singleBlock=0, levelsProcessed=0, overflow=0
currLevel=10, currSize=113798, nextSize=141175, use_singleBlock=0, levelsProcessed=0, overflow=0
currLevel=11, currSize=141175, nextSize=120940, use_singleBlock=0, levelsProcessed=0, overflow=0
currLevel=12, currSize=120940, nextSize=68631, use_singleBlock=0, levelsProcessed=0, overflow=0
currLevel=13, currSize=68631, nextSize=33313, use_singleBlock=0, levelsProcessed=0, overflow=0
currLevel=14, currSize=33313, nextSize=16294, use_singleBlock=0, levelsProcessed=0, overflow=0
currLevel=15, currSize=16294, nextSize=7537, use_singleBlock=0, levelsProcessed=0, overflow=0
currLevel=16, currSize=7537, nextSize=3559, use_singleBlock=0, levelsProcessed=0, overflow=0
currLevel=17, currSize=3559, nextSize=1654, use_singleBlock=0, levelsProcessed=0, overflow=0
currLevel=18, currSize=1654, nextSize=791, use_singleBlock=0, levelsProcessed=0, overflow=0
currLevel=19, currSize=791, nextSize=298, use_singleBlock=0, levelsProcessed=0, overflow=0
currLevel=20, currSize=298, nextSize=224, use_singleBlock=0, levelsProcessed=0, overflow=0
currLevel=21, currSize=224, nextSize=0, use_singleBlock=1, levelsProcessed=12, overflow=0
Visited vertices: 600493
Max levels: 32
```