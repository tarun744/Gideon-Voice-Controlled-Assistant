import speech_recognition as sr
import os
import pyttsx3
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import playsound
import tensorflow
import random
import json
from selenium.webdriver.common.keys import Keys
import pickle
import webbrowser
import re
from selenium.webdriver.common.action_chains import ActionChains

import smtplib
import requests
import subprocess
from pyowm import OWM
import datetime
from selenium import webdriver
 
import youtube_dl
# import vlc
import pyjokes
import urllib.request
# import urllib
# import urllib2
from selenium.webdriver.common.by import By

import json
import winshell
import datetime
import time

from bs4 import BeautifulSoup as soup
# from urllib2 import urlopen
import wikipedia
import random
from time import strftime
with open("intents.json") as file:
    data = json.load(file)
import os
import subprocess

import ctypes
from GoogleNews import GoogleNews
import shutil


# # Initialize the recognizer
# r = sr.Recognizer()
# engine = pyttsx3.init('sapi5')
# # Function to convert text to
# # speech

# voices = engine.getProperty('voices')  # getting details of current voice
# engine.setProperty('voice', voices[1].id)
# engine.say("Sir I'm glad you are here")
# engine.say("What can i do for u ?")


# def SpeakText(command):
#     # Initialize the engine

#     engine.say(command)
#     engine.runAndWait()


# # Loop infinitely for user to
# # speak

# while (1):

#     # Exception handling to handle
#     # exceptions at the runtime
#     try:


#         # use the microphone as source for input.
#         with sr.Microphone() as source2:

#             # wait for a second to let the recognizer
#             # adjust the energy threshold based on
#             # the surrounding noise level

#             r.dynamic_energy_threshold = True
#             r.dynamic_energy_adjustment_damping = 0.1


#             r.energy_threshold = 350
#             r.pause_threshold = 0.5
#             r.adjust_for_ambient_noise(source2, duration=0.4)
#             # listens for the user's input

#             audio2 = r.listen(source2)

#             # Using ggogle to recognize audio
#             MyText = r.recognize_google(audio2, language='en-IN')
#             MyText = MyText.lower()

#         if 'turn off gideon' in MyText:
#             SpeakText("by by Sir")
#             break
#         else:
#             if 'shuttdown' in MyText:
#                 SpeakText("as u say sir!")
#                 SpeakText("shutting down")
#                 break
#             else:
#                 if 'shutt yourself down' in MyText:
#                     SpeakText("as u say sir!")
#                     SpeakText("shutting down")
#                     break

#                 else:
#                     if 'shutdown' in MyText:
#                         SpeakText("as u say sir!")
#                         SpeakText("turning down")
#                         break
#                     else:
#                         if 'power off' in MyText:
#                             SpeakText("as u say sir!")
#                             SpeakText("Shutting down")
#                             break
#                         else:
#                             if 'shut down' in MyText:
#                                     SpeakText("as u say sir!")
#                                     SpeakText("Shutting down")
#                                     break

#                             else:
#                                 if 'power of' in MyText:
#                                         SpeakText("as u say sir!")
#                                         SpeakText("Shutting down")
#                                         break

#             SpeakText(MyText)
#             print(MyText)

        # if 'open reddit'  in MyText:
        #     reg_ex = re.search('open reddit (.*)', MyText)
        #     url = 'https://www.reddit.com/'
        #     if reg_ex:
        #         subreddit = reg_ex.group(1)
        #         url = url + 'r/' + subreddit
        #     webbrowser.open(url)
        #     SpeakText('The Reddit content has been opened for you Sir.')





#     except sr.RequestError as e:
#         print("Could not request results; {0}".format(e))

#     except sr.UnknownValueError:
#         SpeakText("I'm listening")










try:
    
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)


try:
    model.load("./model.tflearn")
except:

    from tensorflow.python.framework import ops
    ops.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

    model.fit(training, output, n_epoch=1110, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)



def chat():
    r = sr.Recognizer()
    engine = pyttsx3.init('sapi5')
    # Function to convert text to
    # speech

    voices = engine.getProperty('voices')  # getting details of current voice
    engine.setProperty('voice', voices[1].id)
    hour = int(datetime.datetime.now().hour)
    if (hour>= 0 and hour<12):
        engine.say("Good Morning Sir !") 
    else:
        if(hour>= 12 and hour<18):
            engine.say("Good Afternoon Sir !")  
        else:
            engine.say("Good Evening Sir !")


    def SpeakText(command):                # Initialize the engine
            engine.say(command)
            engine.runAndWait()
    

    print("Start talking with the bot (type quit to stop)!")
    while True:
        try:

         # use the microphone as source for input.
            with sr.Microphone() as source2:

                # wait for a second to let the recognizer
                # adjust the energy threshold based on
                # the surrounding noise level

                r.dynamic_energy_threshold = True
                r.dynamic_energy_adjustment_damping = 0.1


                r.energy_threshold = 300
                r.pause_threshold = 0.5
                r.adjust_for_ambient_noise(source2, duration=0.4)
                # listens for the user's input

                audio2 = r.listen(source2)

                # Using ggogle to recognize audio
                MyText = r.recognize_google(audio2, language='en-IN')
                MyText = MyText.lower()
            fgroup
            inp = MyText
            print(inp)
            if inp.lower() == "quit":
                break

            results = model.predict([bag_of_words(inp, words)])
            results_index = numpy.argmax(results)
            tag = labels[results_index]

            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
                    patterns = tg['patterns']

            if 'o' in MyText:
                gn = GoogleNews()
                
                s=gn.get_texts()
                      
                SpeakText(s)
            else:
                if 'stem' in tag:
                    subprocess.Popen(r'explorer /select,"C:\Windows\Temp"')
            
                else:
                    if 'df' in MyText:
                        
                        webbrowser.open(MyText)
              
                    else:
                        if 'search' in MyText:
                            reg_ex = re.search('search (.+)', MyText)
                            if reg_ex:
                                domain = reg_ex.group(1)
                                print(domain)
                                
                                tabUrl = 'http://google.com/search?q='+domain
                                driver = webdriver.Chrome()

      
    # enter keyword to search
                                
      
    # get google.co.in
                                driver.get(tabUrl)
                                print(driver.title)
                                SpeakText('The website you have requested has been opened for you Sir.')
                                
                            else:
                                pass


                    if 'back' in MyText:
                            driver.back()

                    else:
                        if 'web' in tag:
                            try:
                                news_url="https://news.google.com/news/rss"
                                Client=urllib.request.urlopen(news_url)
                                xml_page=Client.read()
                                Client.close()
                                soup_page=soup(xml_page,"xml")
                                news_list=soup_page.findAll("item")
                                for news in news_list[:5]:
                                    newVoiceRate = 145
                                    engine.setProperty('rate',newVoiceRate)
                                    SpeakText(news.title.text.encode('utf-8'))
                                SpeakText('more news')
                                
                            except Exception as e:
                                    print(e)
                          

                                # else:
                                 #     pass
                           
                        else:
                            if 'empty recycle bin' in MyText:
                                winshell.recycle_bin().empty(confirm = False, show_progress = False, sound = True)
                                SpeakText("Recycle Bin Recycled")

                            else:
                                if "lock window" in MyText or "lock" in MyText:
                                    SpeakText("Sleeping bye   bye")
                                    ctypes.windll.user32.LockWorkStation()
                                else:
                                    if "joke" in MyText:
                                        SpeakText(pyjokes.get_joke())
                                        print(pyjokes.get_joke())


                                    
                                        # if "weather" in MyText:                        # Google Open weather website            # to get API of Open weather
                                        #     api_key = "Api key"
                                        #     base_url = "http://api.openweathermap.org / data s/ 2.5 / weather?"
                                        #     SpeakText(" City name ")
                                        #     print("City name : ")
                                        #     city_name = MyText
                                        #     complete_url = base_url + "appid =" + api_key + "&q =" + city_name
                                        #     response = requests.get(complete_url)
                                        #     x = response.json()
                 
                                        # if x["cod"] != "404":
                                        #     y = x["main"]
                                        #     current_temperature = y["temp"]
                                        #     current_pressure = y["pressure"]
                                        #     current_humidiy = y["humidity"]
                                        #     z = x["weather"]
                                        #     weather_description = z[0]["description"]
                                        #     print(" Temperature (in kelvin unit) = " +str(current_temperature)+"\n atmospheric pressure (in hPa unit) ="+str(current_pressure) +"\n humidity (in percentage) = " +str(current_humidiy) +"\n description = " +str(weather_description))
                                  
                                        
                                    else:          
                                        if 'shutdown' in tag:
                                            SpeakText(random.choice(responses))
                                            break
                                        else:
                                            if 'time' in MyText:

                                                now = datetime.datetime.now()
                                                SpeakText('Current time is %d hours %d minutes' % (now.hour, now.minute))
                                            else:
                                                if 'day' in MyText:
                                                    now=datetime.datetime.now()
                                                    SpeakText('Sir its'+ now.strftime("%A") + now.strftime("%d") + now.strftime("%B") + now.strftime("%Y") )
                                                else:
                                                    if 'song' in MyText:
                                                        path = r'C:\Users\Administrator\Documents'
                                                        folder = path
                                                        for the_file in os.listdir(folder):
                                                            file_path = os.path.join(folder, the_file)
                                                            
                                                                
                                                            if os.path.isfile(file_path):
                                                                os.unlink(file_path)
                                                           
                                                            
                                                            SpeakText('What song shall I play Sir?')
                                                            mysong = MyText
                                                        if mysong:
                                                            flag = 0
                                                            url = "https://www.youtube.com/results?search_query=" + mysong.replace(' ', '+')
                                                            response = urllib2.urlopen(url)
                                                            html = response.read()
                                                            soup1 = soup(html,"lxml")
                                                            url_list = []
                                                            for vid in soup1.findAll(attrs={'class':'yt-uix-tile-link'}):
                                                                if ('https://www.youtube.com' + vid['href']).startswith("https://www.youtube.com/watch?v="):
                                                                    flag = 1
                                                                    final_url = 'https://www.youtube.com' + vid['href']
                                                                    url_list.append(final_url)
                                                                    url = url_list[0]
                                                            ydl_opts = {}
                                                            os.chdir(path)
                                                            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                                                                ydl.download([url])
                                                            vlc.play(path)
                                                            if flag == 0:
                                                                SpeakText('I have not found anything in Youtube ')
                                                    else:
                                                        if "greeting" in tag:
                                                            tarun=random.choice(responses) 
                                                            SpeakText(tarun)
                                                            print(tarun)

                                                        else:
                                                            if "goodbye" in tag:
                                                                tarun=random.choice(responses) 
                                                                SpeakText(tarun)
                                                                print(tarun)

                                                            else:
                                                                if "thanks" in tag:
                                                                    tarun=random.choice(responses) 
                                                                    SpeakText(tarun)
                                                                    print(tarun)

                                                                else:
                                                                    SpeakText("sorry sir i didnt get u ")



            # if 'open' in MyText:
            #     reg_ex = re.search('open (.+)', MyText)
            #     if reg_ex:
            #         domain = reg_ex.group(1)
            #         print(domain)
            #         url = 'https://www.' + domain + '.com'
            #         webbrowser.open(url)
            #         SpeakText('The website you have requested has been opened for you Sir.')
            #     else:
            #         pass        


        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

        except sr.UnknownValueError:
            SpeakText("I'm listening")
        
chat()