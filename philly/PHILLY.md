# upload data
```
powershell philly\philly-fs.ps1 -cp -r data-bin/cnndm //philly/eu2/ipgsrch/yushi/fairseq/data-bin
```

# submit job
```
curl -k --ntlm --user : -X POST -H "Content-Type: application/json" --data @job.json https://philly/api/jobs
```

# monitor job
```
curl -k --ntlm --user : "https://philly/api/status?clusterId=eu2&vcId=ipgsrch&jobId=application_1555486458178_13653&jobType=cust&content=full"
```

# abort job
```
curl -k --ntlm --user : "https://philly/api/abort?clusterId=eu2&jobId=application_1450168330223_0011"
```

# list jobs
```
curl -k --ntlm --user : "https://philly/api/list?clusterId=eu2&vcId=ipgsrch"
```

# get job metadata
```
curl -k --ntlm --user : "https://philly/api/metadata?clusterId=eu2&vcId=ipgsrch&jobId=application_1555486458178_13653"
```

# encryption (not work yet)

**create .netrc with the following format**
```
machine philly
login yushi
password <your password>
```

**create a key**
```
gpg --gen-key
```

**encrypt .netrc**
```
gpg -r yushi -e .netrc
```

**remove .netrc**
```
rm .netrc
```

**add decryption in curl command**
```
gpg --batch -q -d .netrc.gpg | curl --netrc-file /dev/stdin -n <url>
```
