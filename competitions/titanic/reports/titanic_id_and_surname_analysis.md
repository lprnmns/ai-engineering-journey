# Titanic PassengerId and Surname Analysis

## Goal

Check two possible hidden patterns:

1. PassengerId order groups
2. Surname / family groups

## PassengerId Bin Survival

| PassengerIdRange | passenger_count | survived_count | survival_rate |
| --- | --- | --- | --- |
| 1-90 | 90 | 39 | 0.43333333333333335 |
| 91-179 | 89 | 20 | 0.2247191011235955 |
| 180-268 | 89 | 34 | 0.38202247191011235 |
| 269-357 | 89 | 45 | 0.5056179775280899 |
| 358-446 | 89 | 37 | 0.4157303370786517 |
| 447-535 | 89 | 32 | 0.3595505617977528 |
| 536-624 | 89 | 39 | 0.43820224719101125 |
| 625-713 | 89 | 33 | 0.3707865168539326 |
| 714-802 | 89 | 30 | 0.33707865168539325 |
| 803-891 | 89 | 33 | 0.3707865168539326 |

## Surname Survival Groups

Only surnames with at least 2 passengers are shown below.

| Surname | passenger_count | survived_count | survival_rate | passenger_ids | names |
| --- | --- | --- | --- | --- | --- |
| Andersson | 9 | 2 | 0.2222222222222222 | 14, 69, 120, 147, 542, 543, 611, 814, 851 | Andersson, Mr. Anders Johan | Andersson, Miss. Erna Alexandra | Andersson, Miss. Ellis Anna Maria | Andersson, Mr. August Edvard ("Wennerstrom") | Andersson, Miss. Ingeborg Constanzia | Andersson, Miss. Sigrid Elisabeth | Andersson, Mrs. Anders Johan (Alfrida Konstantia Brogren) | Andersson, Miss. Ebba Iris Alfrida | Andersson, Master. Sigvard Harald Elias |
| Sage | 7 | 0 | 0.0 | 160, 181, 202, 325, 793, 847, 864 | Sage, Master. Thomas Henry | Sage, Miss. Constance Gladys | Sage, Mr. Frederick | Sage, Mr. George John Jr | Sage, Miss. Stella Anna | Sage, Mr. Douglas Bullen | Sage, Miss. Dorothy Edith "Dolly" |
| Carter | 6 | 4 | 0.6666666666666666 | 250, 391, 436, 764, 803, 855 | Carter, Rev. Ernest Courtenay | Carter, Mr. William Ernest | Carter, Miss. Lucile Polk | Carter, Mrs. William Ernest (Lucile Polk) | Carter, Master. William Thornton II | Carter, Mrs. Ernest Courtenay (Lilian Hughes) |
| Johnson | 6 | 3 | 0.5 | 9, 173, 303, 598, 720, 870 | Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg) | Johnson, Miss. Eleanor Ileen | Johnson, Mr. William Cahoone Jr | Johnson, Mr. Alfred | Johnson, Mr. Malkolm Joackim | Johnson, Master. Harold Theodor |
| Goodwin | 6 | 0 | 0.0 | 60, 72, 387, 481, 679, 684 | Goodwin, Master. William Frederick | Goodwin, Miss. Lillian Amy | Goodwin, Master. Sidney Leonard | Goodwin, Master. Harold Victor | Goodwin, Mrs. Frederick (Augusta Tyler) | Goodwin, Mr. Charles Edward |
| Panula | 6 | 0 | 0.0 | 51, 165, 267, 639, 687, 825 | Panula, Master. Juha Niilo | Panula, Master. Eino Viljami | Panula, Mr. Ernesti Arvid | Panula, Mrs. Juha (Maria Emilia Ojala) | Panula, Mr. Jaako Arnold | Panula, Master. Urho Abraham |
| Skoog | 6 | 0 | 0.0 | 64, 168, 361, 635, 643, 820 | Skoog, Master. Harald | Skoog, Mrs. William (Anna Bernhardina Karlsson) | Skoog, Mr. Wilhelm | Skoog, Miss. Mabel | Skoog, Miss. Margit Elizabeth | Skoog, Master. Karl Thorsten |
| Rice | 5 | 0 | 0.0 | 17, 172, 279, 788, 886 | Rice, Master. Eugene | Rice, Master. Arthur | Rice, Master. Eric | Rice, Master. George Hugh | Rice, Mrs. William (Margaret Norton) |
| Baclini | 4 | 4 | 1.0 | 449, 470, 645, 859 | Baclini, Miss. Marie Catherine | Baclini, Miss. Helene Barbara | Baclini, Miss. Eugenie | Baclini, Mrs. Solomon (Latifa Qurban) |
| Asplund | 4 | 3 | 0.75 | 26, 183, 234, 262 | Asplund, Mrs. Carl Oscar (Selma Augusta Emilia Johansson) | Asplund, Master. Clarence Gustaf Hugo | Asplund, Miss. Lillian Gertrud | Asplund, Master. Edvin Rojj Felix |
| Brown | 4 | 3 | 0.75 | 195, 346, 671, 685 | Brown, Mrs. James Joseph (Margaret Tobin) | Brown, Miss. Amelia "Mildred" | Brown, Mrs. Thomas William Solomon (Elizabeth Catherine Ford) | Brown, Mr. Thomas William Solomon |
| Harper | 4 | 3 | 0.75 | 53, 646, 721, 849 | Harper, Mrs. Henry Sleeper (Myna Haxtun) | Harper, Mr. Henry Sleeper | Harper, Miss. Annie Jessie "Nina" | Harper, Rev. John |
| Kelly | 4 | 3 | 0.75 | 301, 574, 697, 707 | Kelly, Miss. Anna Katherine "Annie Kate" | Kelly, Miss. Mary | Kelly, Mr. James | Kelly, Mrs. Florence "Fannie" |
| Fortune | 4 | 2 | 0.5 | 28, 89, 342, 439 | Fortune, Mr. Charles Alexander | Fortune, Miss. Mabel Helen | Fortune, Miss. Alice Elizabeth | Fortune, Mr. Mark |
| Harris | 4 | 2 | 0.5 | 63, 220, 231, 571 | Harris, Mr. Henry Birkhardt | Harris, Mr. Walter | Harris, Mrs. Henry Birkhardt (Irene Wallach) | Harris, Mr. George |
| Hart | 4 | 2 | 0.5 | 315, 412, 441, 536 | Hart, Mr. Benjamin | Hart, Mr. Henry | Hart, Mrs. Benjamin (Esther Ada Bloomfield) | Hart, Miss. Eva Miriam |
| Smith | 4 | 1 | 0.25 | 175, 261, 285, 347 | Smith, Mr. James Clinch | Smith, Mr. Thomas | Smith, Mr. Richard William | Smith, Miss. Marion Elsie |
| Williams | 4 | 1 | 0.25 | 18, 156, 305, 736 | Williams, Mr. Charles Eugene | Williams, Mr. Charles Duane | Williams, Mr. Howard Hugh "Harry" | Williams, Mr. Leslie |
| Ford | 4 | 0 | 0.0 | 87, 148, 437, 737 | Ford, Mr. William Neal | Ford, Miss. Robina Maggie "Ruby" | Ford, Miss. Doolina Margaret "Daisy" | Ford, Mrs. Edward (Margaret Ann Watson) |
| Gustafsson | 4 | 0 | 0.0 | 105, 380, 393, 877 | Gustafsson, Mr. Anders Vilhelm | Gustafsson, Mr. Karl Gideon | Gustafsson, Mr. Johan Birger | Gustafsson, Mr. Alfred Ossian |
| Lefebre | 4 | 0 | 0.0 | 177, 230, 410, 486 | Lefebre, Master. Henry Forbes | Lefebre, Miss. Mathilde | Lefebre, Miss. Ida | Lefebre, Miss. Jeannie |
| Palsson | 4 | 0 | 0.0 | 8, 25, 375, 568 | Palsson, Master. Gosta Leonard | Palsson, Miss. Torborg Danira | Palsson, Miss. Stina Viola | Palsson, Mrs. Nils (Alma Cornelia Berglund) |
| Richards | 3 | 3 | 1.0 | 408, 438, 832 | Richards, Master. William Rowe | Richards, Mrs. Sidney (Emily Hocking) | Richards, Master. George Sibley |
| Collyer | 3 | 2 | 0.6666666666666666 | 238, 638, 802 | Collyer, Miss. Marjorie "Lottie" | Collyer, Mr. Harvey | Collyer, Mrs. Harvey (Charlotte Annie Tate) |
| Goldsmith | 3 | 2 | 0.6666666666666666 | 166, 329, 549 | Goldsmith, Master. Frank John William "Frankie" | Goldsmith, Mrs. Frank John (Emily Alice Brown) | Goldsmith, Mr. Frank John |
| Graham | 3 | 2 | 0.6666666666666666 | 269, 333, 888 | Graham, Mrs. William Thompson (Edith Junkins) | Graham, Mr. George Edward | Graham, Miss. Margaret Edith |
| Hoyt | 3 | 2 | 0.6666666666666666 | 225, 487, 794 | Hoyt, Mr. Frederick Maxfield | Hoyt, Mrs. Frederick Maxfield (Jane Anne Forby) | Hoyt, Mr. William Fisher |
| Laroche | 3 | 2 | 0.6666666666666666 | 44, 609, 686 | Laroche, Miss. Simonne Marie Anne Andree | Laroche, Mrs. Joseph (Juliette Marie Louise Lafargue) | Laroche, Mr. Joseph Philippe Lemercier |
| Navratil | 3 | 2 | 0.6666666666666666 | 149, 194, 341 | Navratil, Mr. Michel ("Louis M Hoffman") | Navratil, Master. Michel M | Navratil, Master. Edmond Roger |
| Newell | 3 | 2 | 0.6666666666666666 | 216, 394, 660 | Newell, Miss. Madeleine | Newell, Miss. Marjorie | Newell, Mr. Arthur Webster |

## How To Read This

### PassengerId bins

If survival rate changes strongly across PassengerId ranges, there may be ordering/distribution effects.

But be careful:

```text
PassengerId itself should usually not be used as a direct model feature.
It can reveal dataset ordering, but it may not generalize.
```

### Surname groups

If the same surname appears multiple times, it may indicate family members.

This can be useful because families often share:

```text
Ticket
Cabin
Pclass
Fare
Survival conditions
```

Safe feature examples:

```text
SurnameGroupSize
Surname appears in train/test combined
Family group structure
```

Risky feature examples:

```text
Surname target mean survival
Using train survival labels to directly predict test family members
```
