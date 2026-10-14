# split_70_25_5.ps1  –  Perfect 70/15/15 split for your Barako dataset
$source = "F:\codes\goofyprojs\cropImages"
$dest   = "F:\codes\school\prototype\coffee_dataset\robusta" #Change the path for arabica/robusta

# Remove old dataset if exists
if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
mkdir $dest | Out-Null

# Create all folders
foreach ($split in "train", "val", "test") {
    mkdir "$dest\$split" -Force | Out-Null
    foreach ($class in "black", "broken", "moldy") {
        mkdir "$dest\$split\$class" -Force | Out-Null
    }
}

# Mapping old folder → clean class name
# $mapping = @{
#     "cropped_arabica_black"  = "black"
#     "cropped_arabica_broken" = "broken"
#     "cropped_arabica_moldy"  = "moldy"
# }

$mapping = @{
    "cropped_robusta_black"  = "black"
    "cropped_robusta_broken" = "broken"
    "cropped_robusta_moldy"  = "moldy"
}

foreach ($entry in $mapping.GetEnumerator()) {
    $old_folder = Join-Path $source $entry.Key
    $class_name = $entry.Value

    # Get ALL image files
    $images = Get-ChildItem $old_folder -Include *.jpg, *.jpeg, *.png, *.JPG, *.JPEG, *.PNG -Recurse
    $total   = $images.Count

    if ($total -eq 0) {
        Write-Warning "No images found in $old_folder"
        continue
    }

    # Shuffle randomly (reproducible if you want: add -Seed 42)
    $images = $images | Sort-Object { Get-Random }

    # Calculate split indices
    $train_count = [int]($total * 0.70)
    $val_count   = [int]($total * 0.15)
    $test_count  = $total - $train_count - $val_count  # remaining ≈15%

    # Copy to folders
    0..($train_count-1) | ForEach-Object { Copy-Item $images[$_].FullName "$dest\train\$class_name\" }
    $train_count..($train_count+$val_count-1) | ForEach-Object { Copy-Item $images[$_].FullName "$dest\val\$class_name\" }
    ($train_count+$val_count)..($total-1) | ForEach-Object { Copy-Item $images[$_].FullName "$dest\test\$class_name\" }

    Write-Host "$class_name → $total images | train:$train_count  val:$val_count  test:$test_count" -ForegroundColor Cyan
}

# Final summary
Write-Host "`nDataset successfully created with 70/15/15 split!" -ForegroundColor Green
Write-Host "Location: $dest"
Get-ChildItem $dest -Recurse -Directory | ForEach-Object {
    $count = (Get-ChildItem $_.FullName -File).Count
    Write-Host "$($_.FullName.Replace($dest,'')) : $count images" -ForegroundColor Yellow
}