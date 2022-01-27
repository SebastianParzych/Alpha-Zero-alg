# Ogólny algorytm ALpha Zero

Repozytorium przedstawia implementację algorytmu Alpha-Zero dla gier:
- kółko i krzyżyk
- connect 4  

Poza samą implementacją samego algorytmu został dodany moduł do gry z wytrenowanymi modelami do obu gier.
# Instrukcja

## Przygotowanie środowiska  
<br>

1. Sklonowanie repozytorium  
   
```
git clone https://github.com/SebastianParzych/Alpha-Zero-alg.git
```

2. Do uruchomienia projektu należy posiadać zainstalowanego Pythona w wersji >=3.9.6
   
2. Jeżeli nie posiadamy w pip-ie pakietu do tworzenia wirtualnego środowiska do projektów należy wpierw go zainstalować:
```
    pip install virtualenv
```
3. Następnie znajdując się w głównym folderze projektu, tworzymy środowisko do niego: 
```
    virtualenv env 
```
4. Aktywacja środowiska
```
    /env/Scripts/activate
```
5. Zainstalowanie wszystkich wymaganych pakietów z pliku requirements.txt:
```
    pip install -r ./requirements.txt
```
## Jak uruchomić projekt.
   <br>
Są możliwe do wykonania dwie akcje:  
 
<br>

>1. Uczenie algorytmu. 
   <br></br>  

   - Dla gry kółko i krzyżyk:  
  
```
    python main.py --action=train --game=ttt
```
   - Dla gry connect4:  
```
    python main.py --action=train --game=connect4
```
 <br></br>


>2.  Rozegranie Partii z algorytmetm w wybranej grze. 
 <br></br>

- Dla gry kółko i krzyżuk:           
```
    python main.py --action=train --game=ttt --player-mark=x
```
Przy wykonywaniu ruchu użytkownik wpisuje położenie x,y swojego znaku np. `0,0`
- Dla gry connect4:  
```
    python main.py --action=train --game=connect4 --player-mark=o
```
Przy wykonywaniu ruchu użytkownik wpisuje indeks kolumny, do której ma być dołożony znak znak np. `0`
<br>
<br>
Po więcej informacji warto skorzystać z komendty  ```--help```
```
python main.py --help
``` 
```
usage: main.py [-h] [--action [ACTION]] [--game [GAME]] [--player-mark [PLAYER_MARK]]

optional arguments:
  -h, --help            show this help message and exit
  --action [ACTION]     [train/play] to train or play with model
  --game [GAME]         [connect4/ttt] to pick specific game connect4 or tictactoe   
  --player-mark [PLAYER_MARK]
                        [x/X/o/O] to pick specific side of game
```
## Przykładowe uczenie i logging:
```
2022-01-27 12:32:55,021: INFO     Starting Training Pipeline...
2022-01-27 12:32:55,021: INFO     TrainConfig(bundle_count=2, games_per_bundle=8, batch_size=16, epochs_per_iteration=12, evalgames_per_iteration=10)
2022-01-27 12:32:55,022: INFO     Loading existing 2 data from last iteration
2022-01-27 12:32:55,029: INFO     2 size of generated data
2022-01-27 12:32:55,029: INFO     Starting current 0 training iteration
2022-01-27 12:32:55,030: INFO     Training total iteration 13
2022-01-27 12:32:55,046: INFO     Prepearing training dataset ... 
2022-01-27 12:32:55,047: INFO     Prepearsed samples: (192, 192, 192) states,policies,values
2022-01-27 12:32:55,048: INFO     Augmented data: (1536, 1536, 1536)
2022-01-27 12:32:55,049: INFO     Train model, 12 epochs
Epoch 1/12
96/96 [==============================] - 3s 4ms/step - loss: 0.5897 - policy_head_loss: 0.8901 - value_head_loss: 0.2092 - policy_head_accuracy: 0.7611 - value_head_accuracy: 0.7070
```
## Przykładowa Gra
