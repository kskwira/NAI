<h3>
Authors: Krzysztof Skwira & Tomasz Lemke
</h3>
Aim of this project is to "teach" the Computer the best moves to win a selected game.
The selected game is from the classic retro arcade - Ms. Packman.
By using the created game environment and the Reinforced Learning algorithms we teach the AI by making random moves and recording the scores based on the decision made.
With each iteration the AI can take into consideration the previous outcomes of the step taken and with the help of MinMax algorithm select the best move in the current situation.

For the purpose of this game have used an adjusted version of the Q-learning algorithm formula.

<h3>
Installation: 
</h3>

pip install gym[all] \
pip install gym[atari,accept-rom-license] \
pip install numpy


<h3>
Reference:
</h3>

https://en.wikipedia.org/wiki/Reinforcement_learning \
https://en.wikipedia.org/wiki/Q-learning \
https://en.wikipedia.org/wiki/Ms._Pac-Man