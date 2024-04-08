# Check if tar, curl, and Get-FileHash (equivalent to md5sum) commands are available
if ((Get-Command tar -ErrorAction SilentlyContinue) -eq $null -or (Get-Command curl -ErrorAction SilentlyContinue) -eq $null -or (Get-Command Get-FileHash -ErrorAction SilentlyContinue) -eq $null) {
    Write-Host "tar or curl or Get-FileHash commands not found"
    Write-Host ""
    Write-Host "Please, install it first."
    Write-Host "If you cannot/dontwantto install these, you may download the features manually."
    Write-Host "You may find the links and correct paths in the repo README."
    Write-Host "Make sure to check the md5 sums after manual download."
    Write-Host "Extraction commands can be checked in this file."
    exit
}

function download_check_expand_rmtar {
    param($BASE_LINK, $FNAME, $WHERE_TO, $MD5SUM_GT)
    Write-Host "Downloading $FNAME"
    curl $BASE_LINK/$FNAME --create-dirs -o $WHERE_TO/$FNAME
    Write-Host "Checking tar md5sum"
    $md5 = Get-FileHash -Algorithm MD5 $WHERE_TO/$FNAME
    if ((Get-Content $MD5SUM_GT | Select-String -Pattern $FNAME) -ne $md5.Hash) {
        Write-Host "MD5 checksum failed"
        exit
    }
    Write-Host "Expanding tar"
    tar xf $WHERE_TO/$FNAME -C $WHERE_TO
    Write-Host "Removing tar"
    Remove-Item $WHERE_TO/$FNAME
    Write-Host ""
}

$WHERE_TO = "./downloaded_features"
$BASE_LINK = "https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/vas"
$MD5SUM_GT = "./md5sum_vas.md5"

$strings = "dog", "fireworks", "drum", "baby", "gun", "sneeze", "cough", "hammer"
foreach ($class in $strings) {
    # spectrograms
    $FNAME = "${class}_melspec_10s_22050hz.tar"
    download_check_expand_rmtar $BASE_LINK $FNAME $WHERE_TO $MD5SUM_GT

    # BN Inception Features
    $FNAME = "${class}_feature_rgb_bninception_dim1024_21.5fps.tar"
    download_check_expand_rmtar $BASE_LINK $FNAME $WHERE_TO $MD5SUM_GT

    $FNAME = "${class}_feature_flow_bninception_dim1024_21.5fps.tar"
    download_check_expand_rmtar $BASE_LINK $FNAME $WHERE_TO $MD5SUM_GT

    # ResNet50 Features
    # $FNAME = "${class}_feature_resnet50_dim2048_21.5fps.tar"
    # download_check_expand_rmtar $BASE_LINK $FNAME $WHERE_TO $MD5SUM_GT
}
