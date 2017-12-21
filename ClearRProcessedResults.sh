#!/usr/bin/env bash
if [ "$(ls -A RProcessedResults)" ];
    then rm -r RProcessedResults/*
fi