To use this:

(1) download captivates data from Mongo Compass and store it as
'captivatesFiltered.json' in this folder.

(2) npm install (which will install node-static).

(3) make sure you have global browserify installed (npm install -g browserify)

(4) make sure you can run the shell script ("chmod 777 run.sh").

(5) run the shell script "./run.sh", which browserify's the buffer library for
us so we can use it in the browser, and minifies it (along with our chart.js
file) into 'bundle.js', which we include in our chart.html, served by
node-static.

(6) open 'http://localhost:8080/chart.html' in the browser.
