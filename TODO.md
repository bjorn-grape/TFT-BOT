# TODO list and ideas

## Map representation
* 4 x 7 grid
* 9 champions idle

## Stages

Can chat at all stages.

### 0. Start/Finish phase
Press quit button to quit the game.

### 1. Carousel phase
* Create priority list, based on champions and items
* Move towards the chosen champion

### 2. Idle phase
* Observe all players
   * Get champions info (position, items, name)
   * Get player's composition
* Place champions
* Place items
* Buy champions
* Level up
* Reroll

### 3. Combat phase
1. Beginning of fight:
    * Get position of champions on the grid and their items
2. During fight:
    * Track champions' positions with life bar
3. End of fight:
    * Get champions' damages
    * Get winner/loser of each fight
    * Collect items on grid
    
Restart to Carousel phase until end of game.


## Image detection

Based on pattern matching. Riot provides the champions' icons.
The camera is still, so interface elements and the grid are still, so we can draw bounding box for each element.

### Champion detection on the grid
1. Right click on the champion to get description.
2. Get difference between last frame and current frame to get only the champion description
3. Apply text detection and image detection to get champion's name and items

## Meta
We could grab the best composition on external sites or apps like Blitz, in order to tell \\
the AI which champions or compositions to favour