#! /bin/bash
browserify chart.js -o bundle.js
node static-server
