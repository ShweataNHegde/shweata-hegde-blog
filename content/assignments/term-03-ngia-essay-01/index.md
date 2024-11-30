---
author: ["Shweata N. Hegde"]
title: "Term 03 | The New Geography of Information Age | Extension Notes 01:  How Do They Know All About Me?" 
date: "2024-11-04"
description: ""
tags: ["assignments", "term-03"]
ShowToc: true
---
**Understanding Big Data Analysis Behind Recommender Systems and Other Algorithms**

Despite friends discussing targeted advertising and giant companies tracking online activities, I had not bothered learning more about cookies or user profiling. I knew enough to ensure I checked the “Reject All” option whenever I visited a website, but I did not know the implications of not doing so. As Prof. Debayan explained how ad companies such as DoubleClick track me across websites to gather insights about my preferences and thereby show relevant advertisements, I became aware of my ignorance. I also realized that targeted advertising involves analyzing huge amounts of data and became curious to learn more about it.

To get started, I read a book called *Big Data: A Very Short Introduction* by Dawn E. Holmes. The initial chapters of the book cover how big data is stored and analyzed and lead up to the chapter titled “Big Data, Big Business.” Google, Amazon, and Netflix are used as case studies to discuss the Pay-Per-Click model, targeted advertising, and recommender systems (collaborative filtering and cosine similarity). The introductory chapters on storing unstructured data using the Hadoop Distributed File System and analysis methods such as MapReduce, PageRank, and Bloom Filters made me appreciate the complex and innovative algorithms built on top of these. If interested, I would recommend the book to learn more about these concepts, as I won’t be able to explain them in detail in these Extension Notes.

In particular, I was fascinated by the logic behind collaborative filtering, whereby an e-commerce product is recommended based on the purchases of other users whose profiles are similar to mine. This is done by using Jaccard’s index, which is defined as the number of items two sets (customers) have in common, divided by the total number of distinct items.

Let’s say I am interested in cookbooks and have bought *Daily Salads* and *Desserts Tomorrow* from Amazon (Table 1). The collaborative filtering algorithm calculates Jaccard’s index for all the other cookbook buyers by comparing our purchases to recommend what to buy next.  


|                  | Daily Salads | Pasta Today | Desserts Tomorrow | Wine For All |
|------------------|--------------|-------------|-------------------|--------------|
| **Me**           | bought       |             | bought            |              |
| **Jones**        | bought       |             |                   | bought       |
| **Brown**        |              |             | bought            | bought       |

*Table 1: Books Bought by Me, Jones, and Brown (from *Big Data: A Very Short Introduction*)*


| Comparison          | Number of Titles in Common | Total Number of Distinct Titles Purchased | Jaccard Index |
|---------------------|----------------------------|------------------------------------------|---------------|
| **Me and Jones**     | 1                          | 3                                        | 1/3           |
| **Me and Brown**     | 1                          | 4                                        | 1/4           |
| **Jones and Brown**  | 1                          | 4                                        | 1/4           |

*Table 2: Jaccard’s Index (from *Big Data: A Very Short Introduction*)*

We see that the Jaccard’s index is highest between Me and Jones. The book Jones purchased, which I have not, is *Wine For All*, and that is what Amazon recommends (Table 2). Cosine similarity is similar to collaborative filtering; however, it also takes into account the ratings given by customers for the products.

After reading about all the different algorithms in the *Big Data* book, I paused to consider the implications of algorithms. Combined with my reading about the “Warehouse Wage Slave” experience (McClelland), it made me think that we are all at the mercy of a handful of giant companies. We are succumbing to algorithms like recommender systems, prompting these companies to develop even more complex algorithms to further exploit the ever-increasing big data we continue to contribute. We buy more and more products, leading to more and more warehouse workers robotically scanning and packing the products, like cookbooks the algorithm recommended. It is a vicious cycle. Will we even have control over our desires in the future? Or will they be algorithm-driven? Are we becoming robots? I cannot help but worry about the societal implications of these technologies.

This quick research has made me more mindful of the trail I leave on the web. Maybe it is time to move away from big companies like Google and Amazon and do more than just “Reject All.”

**Works Cited** 

1. Holmes, Dawn E. *Big Data: A Very Short Introduction*. Oxford University Press, 2017.  
2. McClelland, Mac. “I Was a Warehouse Wage Slave.” *Mother Jones*, Apr. 2012, www.motherjones.com/politics/2012/02/mac-mcclelland-free-online-shipping-warehouses-labor/