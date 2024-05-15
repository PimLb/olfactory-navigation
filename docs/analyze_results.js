function analyze_csv() {
    const csvFile = document.getElementById("csvFileInput");

    const input = csvFile.files[0];
    const reader = new FileReader();

    console.log('YE')

    reader.onload = function (e) {
        console.log('Bla')
        // var doesColumnExist = false;
        // var data = d3.csv.parse(reader.result, function(d){
        //   doesColumnExist = d.hasOwnProperty("x");
        //   return d;   
        // });
        // console.log(doesColumnExist);
    };
    reader.readAsText(input);

}