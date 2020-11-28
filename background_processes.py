import time

def trackTimeSpent():
	start = time.time()

	file = open("time.txt", "r")
	seconds = float(file.read())
	end = time.time()
	file.close()

	file = open("time.txt", "w")
	seconds += end - start
	file.write(str(seconds))
	file.close()

	start = time.time()

try:
	while True:
		trackTimeSpent()

except KeyboardInterrupt:
	pass