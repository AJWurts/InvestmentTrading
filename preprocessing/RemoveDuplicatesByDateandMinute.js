var databases = ['AAPL', 'ADBE', 'AMD', 'AMZN', 'BAC', "BRK.B", "COST", "DNKN", "GOOG", "HD", "LULU", "MSFT", "NVDA", "SPY", "TGT", "TSLA", "UNH", "VOO", "VTI"];
var array;
var toDelete;
for (var i = 0; i < databases.length; i++) {
    var result = db.getCollection(databases[i]).aggregate([
           {$group: {
               _id: {date: '$date', minute: '$minute'},
               id: {$first: '$_id'},
               count: {$sum: 1}
                }
            }, 
           {$match: {count: {'$gt': 1}}},
           {$group: {
               _id: null,
               toDelete: {$push: '$id'}
           }
       }
       ]);
       
    print(result.hasNext());

}
