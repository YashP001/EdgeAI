# Project overview

This is a summary of our Edge AI Module 1 project, for anyone who doesn't want to go through the full README. The detailed numbers and plots are in ASSIGNMENT_RESULTS.md and the results folder.

## The problem

A doorbell company wants to show users how many days of battery they have left, instead of just a "Low battery" warning. We had to train a small neural network for this, serve it as a live API, and check whether it could run on the doorbell itself.

## The data

We were given 2000 rows of doorbell data in data/team_04_train.csv. Each row has five readings: battery voltage, temperature, rings per day, wifi signal strength (RSSI) and battery age in months. It also has the actual number of days left, which is what we're trying to predict.

When we went through the data we found a few readings that don't make physical sense: 10 rows with zero voltage, 5 with an impossible temperature value and 9 with a wifi signal of exactly 0. We flagged these in row_audit.csv but decided to keep all the rows. We split the data into 1600 rows for training and 400 for validation.

## Baselines

Before training a network we wanted to know what a simple approach gets, so we had something to compare against.

- Always guessing the average days left: off by about 42.7 days on average
- Linear regression: about 25.9 days. It did worst on batteries with a lot of life left, and in one case it was off by more than 100 days.
- The instructor's starter network, without any changes: about 9.1 days

From the plots it was clear that battery life doesn't depend on voltage in a straight line, which is why linear regression struggles here.

## Training the network

We tried hidden layer widths of 4, 8, 16, 32 and 64, each with one and with two hidden layers, so 10 networks in total. We also trained one without normalising the inputs to see how much difference normalisation makes. Every run was trained for 1500 epochs on the same split.

Some things we noticed:

- Normalisation matters a lot. The same 32 -> 32 network had an error of 11.5 days without it and 3.8 days with it.
- Two hidden layers worked much better than one. All the single layer networks stayed at around 10 to 12 days of error, whatever the width.
- Bigger networks helped, but the gain got smaller. Going from 32 -> 32 to 64 -> 64 only improved things slightly.

Our final model is 5 -> 64 -> 64 -> 1 with ReLU and has 4609 parameters. Its validation error is about 3.7 days (training error about 3.3 days). The best validation score came at epoch 334. After that, training error kept dropping but validation error got worse, which means it was starting to overfit, so we saved the epoch 334 version.

## The API

The model is served with FastAPI on GitHub Codespaces. It has these routes:

- /health returns the status, our team number, the parameter count and the validation error
- /model_info shows each layer's parameters, memory size and MACs, and whether the model fits in the doorbell's memory
- /predict takes one reading and returns the predicted days left
- /predict_batch takes a list of readings and returns a prediction for each one
- /docs is a page where you can try out the API in the browser

We set valid ranges for each input based on what a real sensor can report, not just the min and max values in our data. Anything outside those ranges, like a negative voltage or a temperature of 500 degrees, gets a 422 error instead of a prediction. Every response also includes timings for each step (parse, normalise, inference, build and total).

## How fast it is

We used the instructor's bench.py script to time the API from a laptop and from inside the Codespace.

- The model itself takes only about 0.2 ms per prediction.
- From a laptop, a full request takes around 54 ms, so almost all the time goes to the network and not the model.
- With keepalive turned off, every request has to open a new connection and requests took around 500 ms.
- Running the benchmark inside the Codespace on localhost took only about 2 ms, because there's no internet in between.
- Sending 400 readings as one batch request was around 250 times faster than sending them one at a time from the laptop.

## Would it fit on the doorbell?

The doorbell uses an ESP32-S3 chip with about 300 KB of SRAM free for our program. Our model's weights take about 18 KB in float32, or about 4.6 KB if they were stored as int8, so it fits easily either way.

We also worked out the biggest network of the form 5 -> w -> w -> 1 that would still fit. In float32 the largest width that fits is 273, and in int8 it's 550. We estimated one prediction would take less than half a millisecond on the chip, but we didn't test this on a real board.

We think running the model on the doorbell itself makes more sense than calling a server. There's no network delay, the data stays on the device, there are no server costs, and it still works if the wifi goes down. A server does make it easier to update the model, but for this assignment the cloud API is what gets graded.

## Running it

To start the API in the Codespace, run `bash tools/start_codespace.sh` and then set port 8000 to Public in the Ports tab. To retrain the model, run `python train.py --team 4 --sweep --epochs 1500`.

## Team

Team 4 - EdgeX: Swayam Goyal, Yash Bhalotia, Yash Purwar, Priyanshi Gupta
