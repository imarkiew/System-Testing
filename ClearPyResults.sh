#!/usr/bin/env bash
if [ "$(ls -A PyResults)" ];
    then rm -r PyResults/*
fi