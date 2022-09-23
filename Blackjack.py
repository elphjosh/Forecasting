# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 11:24:45 2021

@author: joshe
"""

from IPython.display import clear_output
import random
import sys

suits = ('Hearts', 'Diamonds', 'Spades', 'Clubs')
ranks = ('Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King', 'Ace')
values = {'Two':2, 'Three':3, 'Four':4, 'Five':5, 'Six':6, 'Seven':7, 'Eight':8, 
            'Nine':9, 'Ten':10, 'Jack':10, 'Queen':10, 'King':10, 'Ace':11}

class Card:
    
    def __init__(self,suit,rank):
        self.suit = suit
        self.rank = rank
        self.value = values[rank]
        
    def __str__(self):
        return self.rank + ' of ' + self.suit
    
    
class Deck:
    
    def __init__(self):
        self.all_cards = [] 
        for suit in suits:
            for rank in ranks:
                self.all_cards.append(Card(suit,rank))
                
    def __str__(self):
        deck_comp = ''
        for card in self.all_cards:
            deck_comp += '\n '+card.__str__()
        return 'The deck has:' + deck_comp
                
    def shuffle(self):
        random.shuffle(self.all_cards)
        
    def deal_one(self):
        return self.all_cards.pop()  
    
    
class Player_acc:
    
    def __init__(self,balance):
        self.balance = balance
        
    def lose(self,bet):
        if bet > self.balance:
            print('Funds Unavailable!')
        else:
            print('Bet Accepted')      
            self.balance=self.balance-bet
                  
    def win(self,bet):
        print('Funds Added')      
        self.balance=self.balance+bet+bet
        
    def draw(self,bet):      
        self.balance=self.balance+bet
                  
    def __str__(self):
        return 'Account balance: $'+str(self.balance)
    
    
    
class hand:
    
    def __init__(self):
        self.cards = []
        self.value = 0 
        self.aces = 0 
        self.adjust = 0
        
    def add_card(self,card):
        self.cards.append(card)
        self.value += values[card.rank]
        if values[card.rank] == 11:
            self.adjust = self.adjust + 1
    
    def adjust_for_ace(self):
        for card in self.cards:
            if self.adjust > 0 and self.value > 21:
                self.value = self.value - 10
                self.adjust=self.adjust - 1
                
                
                
def take_bet():
    while True:
        try:
            bet = int(input('How much whould you like to bet?: '))
        except:
            print('Please enter a vaild amount')
        else:
            if bet > account.balance:
                print(f'Sorry, this amount exeeds your maximum balance of ${account.balance} ')
            else:
                return bet
                break
                
    
    
def hit(deck,hand):
    
    hand.add_card(deck.deal_one())
    print(hand.cards[-1])
    
    
def hit_or_stand(deck,hand):
    
    while True:
    
        x = (input('Would you like to hit or Stand? Enter (H)it or (S)tay: ').upper())

        if x.startswith('H'):
            hit(deck,hand)
            return True

        elif x.startswith('S'):
            print("Player stands. Dealer is playing.")
            return False
            break

        else:
            print('Sorry please try again')
            continue
            
        break
    
    
def show_some(player,dealer):
    print(f'\nBet is:${bet}')
    print("Dealer's Hand:")
    print(" <card hidden>")
    print('',dealer.cards[1])  
    print("\nPlayer's Hand:", *player.cards, sep='\n ')
    print("Player's Hand =",player.value)
    
def show_all(player,dealer):
    print(f'\nBet is:${bet}')
    print("Dealer's Hand:", *dealer.cards, sep='\n ')  #same as for loop printing each item in dealer.cards
    print("Dealer's Hand =",dealer.value)
    print("\nPlayer's Hand:", *player.cards, sep='\n ')
    print("Player's Hand =",player.value)
    
    
def player_busts(player,dealer,Player_acc):
    clear_output()
    show_all(player,dealer)
    print("\nPlayer busts!")
    print(account)
    

def player_wins(player,dealer,Player_acc):
    clear_output()
    show_all(player,dealer)
    print("\nPlayer wins!")
    Player_acc.win(bet)
    print(account)

def dealer_busts(player,dealer,Player_acc):
    clear_output()
    show_all(player,dealer)
    print("\nDealer busts!")
    Player_acc.win(bet)
    print(account)
    
def dealer_wins(player,dealer,Player_acc):
    clear_output()
    show_all(player,dealer)
    print("\nDealer wins!")
    print(account)

    
def push(player,dealer,Player_acc):
    clear_output()
    show_all(player,dealer)
    print("\nDealer and Player tie! It's a push.")
    Player_acc.draw(bet)
    print(account)
    
    
    
while True:    
    try:
        account = Player_acc((int(input('How much money would you like to deposit?'))))
        break
    except:
        print('Please enter a valid amount')
        
game = True

while game == True:
    print('Welcome to Blackjack')
    print(account)

    
    
    deck = Deck()
    
    deck.shuffle()
    
    player = hand()
    
    dealer = hand()
    
    if int(account.balance) <= 0:
        print('Go home, you are broke')
        game = False
        sys.exit()
    else:
        pass
        
    player.add_card(deck.deal_one())
    dealer.add_card(deck.deal_one())
    player.add_card(deck.deal_one())
    dealer.add_card(deck.deal_one())
    
    bet=take_bet()
    account.lose(bet)

    
    show_some(player,dealer)
    
    bust = False
    playing = True
    while playing == True:  
        
        playing=hit_or_stand(deck,player)
        
        player.adjust_for_ace()
        
        show_some(player,dealer)
        
        print(player.value)
        
        
            
            
        if player.value > 21:
            player_busts(player,dealer,account)
            playing = False
            bust = True
            break
        
        else:
            continue
 
    dealing = True
    while dealing == True and bust == False:
        
        dealer.adjust_for_ace()
            
        if dealer.value < 17:
            hit(deck,dealer)
            show_all(player,dealer)
    
            
        elif dealer.value > 21:
            dealer_busts(player,dealer,account)
            playing = False
            dealing = False
            bust = True
            break
            
        else:
            dealing = False
            break
            
            
    
    
    
    while bust == False:
        if dealer.value > player.value:
            dealer_wins(player,dealer,account)
            playing = False
            break

        elif dealer.value<player.value:
            player_wins(player,dealer,account)
            playing = False
            break
        else:
            push(player,dealer,account)
            playing = False
            break
    
    while True:
        q=(input('Would you like to play again?(Y or N)')).upper()
        if q.startswith('Y'):
            break
        elif q.startswith('N'):
            print(f'You leave with ${account.balance} ')
            game = False
            break
   
        
    