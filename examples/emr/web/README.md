# Example website

## Steps

1. Follow the steps in https://jekyllrb.com/docs/ to install Jekyll if you do not have.

2. Go to `./web`
```
cd ./web
```

3. Install bundles
```
bundle install
```

4. Start the server
```
bundle exec jekyll serve
```
You may want to add `-H 0.0.0.0` to specify binding to all IPv4 address, and `-P [port]` to specify the port to use.
```
bundle exec jekyll serve -H 0.0.0.0 -P [port]
```
