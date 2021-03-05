const DATA_MAX_DISPLAY = 500;
const HOP_SIZE = 4;
const UPDATE_RATE = 2;

function printKeys(obj){
    for (k in obj) console.log(k);
}

function printTimestamps(obj, num=null){
    if (num==null) { num = obj.length; }

    for (var i=0; i<num;i++){
        console.log(obj[i].packet_tick_ms);
    }
}

function printNovelTimestamps(obj, num=null){
    if (num==null) { num = obj.length; }

    var prev=null;
    for (var i=0; i<num;i++){
        if (obj[i].packet_tick_ms != prev){
            console.log(i + ': ' + obj[i].packet_tick_ms);
            prev = obj[i].packet_tick_ms;
        } else if (! (i % 1000)){
            console.log(i + ' packets reviewed');
        }

    }
}

function extractBlinkVals(obj) {
    var myBuffer = [];

    var buffer = new Buffer(obj.blink_data, 'utf16le');
    for (var i = 0; i < buffer.length; i++) {
        myBuffer.push(buffer[i]);
    }

    return myBuffer.filter((element, index) => {
        return index % 2 === 0;
    })
}

function grabAllBlinkVals(obj){
    return obj.map((o) => { return extractBlinkVals(o);}).flat();
}

const loadJSON = (jsonFile, callback) => {
    let xobj = new XMLHttpRequest();
    xobj.overrideMimeType("application/json");
    xobj.open('GET', jsonFile, true);
    xobj.onreadystatechange = () => {
        if (xobj.readyState === 4 && xobj.status === 200) {
            callback(xobj.responseText);
        }
    };
    xobj.send(null);
}

function sortAndGrabNovelBlinks(obj){
    //step 1: extract timestamp and blink data, push to array of dicts if novel
    var uniqueTimestamps = new Set();
    var novelData = [];

    console.log('starting pruning');

    for (var i=0;i<obj.length;i++){
        //not repeated timestamp, there are thousands of repeated in data
        if (!uniqueTimestamps.has(obj[i].packet_tick_ms)){
            //add it to set of seen data
            uniqueTimestamps.add(obj[i].packet_tick_ms);
            //push it to array
            novelData.push({'ts': obj[i].packet_tick_ms, 'data': extractBlinkVals(obj[i])});
        }
    }

    console.log('original data had ' + obj.length + ' datapoints.');
    console.log('extracted ' + novelData.length + ' novel datapoints.');

    //step 2: sort by timestamp
    novelData.sort(function(first, second) {
            return second.ts - first.ts;
    });

    //step 4: reduce blink data and return
    return novelData.map(el => el.data).flat();

}

function updateChart(chart, all_data, index){
    chart.update({ series: [all_data.slice(index, DATA_MAX_DISPLAY+index)]});

    index += HOP_SIZE;
    if (index + DATA_MAX_DISPLAY >= all_data.length){ index=0; }

    setTimeout(() => { updateChart(chart, all_data, index) }, UPDATE_RATE);

}

const init = () => {
    loadJSON('captivateFiltered.json', (response) => {
        let obj = JSON.parse(response);

        //var bytes = extractBlinkVals(obj[0]);
        //console.log(bytes.length);
        //console.log(bytes);
        //console.log('---');

        //printNovelTimestamps(obj);

        var all_data = sortAndGrabNovelBlinks(obj).slice(0,10000);

        const options = {width: document.width, height: 400, showArea: true,
                        showPoint: false, fullWidth: true, low:0, high:256};
        const data = { series: [all_data.slice(0, DATA_MAX_DISPLAY)] };
        const chart = new Chartist.Line('.ct-chart', data, options);

        setTimeout(() => { updateChart(chart, all_data, HOP_SIZE) }, UPDATE_RATE);

    });
}

window.onload = function(){
    init();
}
