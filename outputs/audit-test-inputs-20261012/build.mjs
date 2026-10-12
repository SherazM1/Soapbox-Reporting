import fs from 'node:fs/promises';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';
const outputDir = "C:/IDocuments/SoapBoxRetail/Soapbox-Reporting/outputs/audit-test-inputs-20261012";
const tests = [
 {name:'photography_cameras',client:'Canon',competitor:'Nikon',query:'DSLR camera',clientPdp:'https://www.walmart.com/ip/245772273',competitorPdp:'https://www.walmart.com/ip/566604061'},
 {name:'office_permanent_markers',client:'Sharpie',competitor:'BIC',query:'permanent markers',clientPdp:'https://www.walmart.com/ip/258144696',competitorPdp:'https://www.walmart.com/ip/19530357'},
 {name:'safety_carbon_monoxide_alarms',client:'Kidde',competitor:'First Alert',query:'carbon monoxide alarm',clientPdp:'https://www.walmart.com/ip/2720795611',competitorPdp:'https://www.walmart.com/ip/17616356915'}
];
for(const t of tests){
 const wb=Workbook.create();const sh=wb.worksheets.add('Sheet1');
 const search='https://www.walmart.com/search?q='+encodeURIComponent(t.query);
 const brand=b=>'https://www.walmart.com/browse/0?facet=brand%3A'+encodeURIComponent(b);
 const rows=[['URL','Evidence Type','Role','Brand'],
 [t.clientPdp,'PDP','Client',t.client],
 [search,'Search','Current',t.client],
 [brand(t.client),'Brand Shop','Client',t.client],
 [t.competitorPdp,'PDP','Competitor',t.competitor],
 [search,'Search','Benchmark',t.competitor],
 [brand(t.competitor),'Brand Shop','Competitor',t.competitor]];
 sh.getRange('A1:D7').values=rows;
 sh.getRange('A1:D7').format.font={name:'Arial',size:12};
 sh.getRange('A1:D1').format.fill='#E7E6E6';
 sh.getRange('A1:D1').format.font={name:'Arial',size:12,bold:true};
 sh.getRange('A1:A7').format.columnWidth=25.375;
 sh.getRange('B1:B7').format.columnWidth=22.625;
 sh.getRange('C1:C7').format.columnWidth=24.375;
 sh.getRange('D1:D7').format.columnWidth=19.375;
 sh.getRange('A2:A7').format.font={name:'Arial',size:12,color:'#0563C1',underline:'single'};
 sh.getRange('A2:A7').format.wrapText=true;
 sh.getRange('A2:D7').format.rowHeight=75;
 sh.getRange('A1:D7').format.verticalAlignment='center';
 wb.recalculate();
 const inspected=await wb.inspect({kind:'table',range:'Sheet1!A1:D7',include:'values',tableMaxRows:7,tableMaxCols:4});
 console.log(t.name,inspected.ndjson);
 const image=await wb.render({sheetName:'Sheet1',range:'A1:D7',scale:1.4,format:'png'});
 await fs.writeFile(outputDir+'/'+t.name+'.png',new Uint8Array(await image.arrayBuffer()));
 await (await SpreadsheetFile.exportXlsx(wb)).save(outputDir+'/'+t.name+'.xlsx');
}
