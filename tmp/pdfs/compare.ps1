$d=Get-Content config/style_guides/sportsoutdoors.json -Raw | ConvertFrom-Json
$lines=Get-Content tmp/pdfs/sports-raw.txt
$sections=@(); $current=$null
foreach($line in $lines){
 $s=$line.Trim()
 if($s -eq 'All Sport Accessories'){break}
 if($s -match '^(.+?) (\d+)$'){
  $current=[pscustomobject]@{name=$matches[1];count=[int]$matches[2];products=[System.Collections.Generic.List[string]]::new()}; $sections += $current
 }elseif($current -and $s){$current.products.Add($s)}
}
function norm($s){($s.ToLower() -replace '[^a-z0-9]','')}
$report=[System.Collections.Generic.List[string]]::new()
$report.Add('Source families: '+$sections.Count+'; Source product count: '+(($sections | Measure-Object count -Sum).Sum))
foreach($sec in $sections){
 $f=@($d.families.PSObject.Properties | Where-Object {(norm $_.Value.display_name) -eq (norm $sec.name)})
 if(!$f){$report.Add('MISSING FAMILY: '+$sec.name+' ('+$sec.count+')');$report.Add('  '+($sec.products -join '; '));continue}
 $ps=@($f[0].Value.product_types.PSObject.Properties)
 $missing=@($sec.products | Where-Object {$name=$_; -not @($ps | Where-Object {(norm $_.Value.display_name) -eq (norm $name)})})
 $extra=@($ps | Where-Object {$name=$_.Value.display_name; -not @($sec.products | Where-Object {(norm $_) -eq (norm $name)})} | ForEach-Object {$_.Value.display_name})
 if($missing.Count -or $extra.Count){$report.Add('DIFFERENCES: '+$sec.name+' source='+$sec.count+' json='+$ps.Count);$report.Add('  Missing: '+($missing -join '; '));$report.Add('  Extra/renamed: '+($extra -join '; '))}
 if($sec.products.Count -ne $sec.count){$report.Add('PARSE COUNT MISMATCH '+$sec.name+' '+$sec.products.Count)}
}
$report | Set-Content tmp/pdfs/section-comparison.txt
$report
