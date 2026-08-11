# EchoRefine Qualitative Examples

## Nepali (npi)

### Accepted Refinement

- Source: The center of Tibetan meditation is the Deity Yoga. Through the visualization of various deities the energy channels are cleaned, the chakras are activated and the enlightenment consciousness is created.
- mBART: Tibetan meditation is the Deity Yoga. Tibetan meditation is the Deity Yoga, Tibetan meditation is the Deity Yoga, Tibetan meditation is the Deity Yoga, Tibetan meditation is the Deity Yoga, Tibetan meditation is the Deity Yoga, Tibetan meditation is the Deity Yoga
- EchoRefine Raw: तिब्बती ध्यान को केन्द्र देवता योग हो । विभिन्न देवताहरूको दृश्यात्मकरण गरेर ऊर्जा च्यानल साफ पारिन्छ, चकरहरू सक्रिय पारिन्छन् र ज्ञान चेतना सिर्जन हुन्छ ।
- EchoRefine Hybrid: तिब्बती ध्यान को केन्द्र देवता योग हो । विभिन्न देवताहरूको दृश्यात्मकरण गरेर ऊर्जा च्यानल साफ पारिन्छ, चकरहरू सक्रिय पारिन्छन् र ज्ञान चेतना सिर्जन हुन्छ ।
- Reference: तिब्बती ध्यानको केन्द्र देवता योग हो। विभिन्न देवताहरूको दृश्यताको माध्यमबाट ऊर्जा च्यानलहरू सफा गरिन्छन्, चक्रहरू सक्रिय हुन्छन् र आत्मज्ञान चेतना सिर्जना हुन्छ।
- Winner: LLM
- Analysis: The QE gate accepted the MBR-selected refinement; sentence-level chrF suggests the accepted edit moved closer to the reference.

### Overcorrection Prevented

- Source: USA Gymnastics supports an independent investigation that may shine light on how abuse of the proportion described so courageously by the survivors of Larry Nassar could have gone undetected for so long and embraces any necessary and appropriate changes.
- mBART: अमेरिका जिम्पसिक्सले लारी नासरको बाँचेकाहरूद्वारा इर्ष्यापूर्वक वर्णन गरिएको अनुपातको दुरुपयोगले कति लामो समयसम्म पत्ता नलगाएको हुन सक्छ भन्नेमा प्रकाश झल्काउन सक्ने एउटा स्वतन्त्र अनुसन्धानलाई समर्थन गर्दछ र आवश्यक र उचित परिवर्तनहरू समाविष्ट गर्दछ ।
- EchoRefine Raw: USA Gymnastics supports an independent investigation that may shine light on how abuse of the proportion described so courageously by the survivors of Larry Nassar could have gone undetected for so long and embraces any necessary and appropriate changes.
- EchoRefine Hybrid: अमेरिका जिम्पसिक्सले लारी नासरको बाँचेकाहरूद्वारा इर्ष्यापूर्वक वर्णन गरिएको अनुपातको दुरुपयोगले कति लामो समयसम्म पत्ता नलगाएको हुन सक्छ भन्नेमा प्रकाश झल्काउन सक्ने एउटा स्वतन्त्र अनुसन्धानलाई समर्थन गर्दछ र आवश्यक र उचित परिवर्तनहरू समाविष्ट गर्दछ ।
- Reference: USA जिमनास्टिक्सले स्वतन्त्र अनुसन्धानलाई समर्थन गर्दछ, जुन लेरी नासारका बचेका मानिसहरूले यति साहसपूर्वक ढंगले वर्णन गरिएको मात्राको दुरुपयोगलाई कसरी यति लामो समयसम्म पत्ता लगाउन नसकेको भन्ने कुरामा प्रकाश पार्न सक्दछ र कुनै आवश्यक र उपयुक्त परिवर्तनलाई अँगाल्दछ।
- Winner: mBART
- Analysis: The raw refinement drifted away from the reference under sentence-level chrF, so the QE gate retained the anchor.

### Anchor Bias Candidate

- Source: The area is also home to an extremely wide variety of animal and bird species.
- mBART: यस क्षेत्रमा धेरै प्रकारका जनावर र पक्षीहरू पनि बस्छन्।
- EchoRefine Raw: यस क्षेत्रले अत्यन्त व्यापक विविधताका जनावर र पन्छी प्रजातिहरूको घर हो।
- EchoRefine Hybrid: यस क्षेत्रमा धेरै प्रकारका जनावर र पक्षीहरू पनि बस्छन्।
- Reference: यो क्षेत्र पशु र चरा प्रजातिहरूको अत्यधिक व्यापक विविधताको घर पनि हो।
- Winner: mBART
- Analysis: The raw refinement scored better than the anchor by sentence-level chrF, but the QE gate retained mBART; this is a candidate anchor-bias case for manual analysis.

## Bengali (ben)

### Accepted Refinement

- Source: The accused are named as Baba Kanjar, Bhutha Kanjar, Rampro Kanjar, Gaza Kanjar and Vishnu Kanjar.
- mBART: অভিজ ্ ঞাকে ডাকা হয় Baba Kanjar, Bhutha Kanjar, Rampro Kanjar, Gaza Kanjar এবং Vishnu Kanjar ।
- EchoRefine Raw: অভিযুক্তদের নাম বাবা কাঞ্জার, ভুটা কাঞ্জার, রামপ্রো কাঞ্জার, গাজা কাঞ্জার ও বিশ্বনাথ কাঞ্জার।
- EchoRefine Hybrid: অভিযুক্তদের নাম বাবা কাঞ্জার, ভুটা কাঞ্জার, রামপ্রো কাঞ্জার, গাজা কাঞ্জার ও বিশ্বনাথ কাঞ্জার।
- Reference: অভিযুক্তদের নাম হল বাবা কাঞ্জার, ভুথা কাঞ্জার, রামপ্রো কাঞ্জার, গাজা কাঞ্জার এবং বিষ্ণু কাঞ্জার।
- Winner: LLM
- Analysis: The QE gate accepted the MBR-selected refinement; sentence-level chrF suggests the accepted edit moved closer to the reference.

### Overcorrection Prevented

- Source: USA Gymnastics supports the United States Olympic Committee's letter and accepts the absolute need of the Olympic family to promote a safe environment for all of our athletes.
- mBART: যুক ্ তরাষ ্ ট ্ রের জিমিস ্ ট ্ যাল কম ্ পিটিমের কাছ থেকে সমর ্ থন পাচ ্ ছে এবং অর ্ থনৈতিক প ্ রয়োজন স ্ কুল বাল ্ পীদের জন ্ য একটি স ্ বাচ ্ ছন ্ ন পরিবেশ প ্ রতিষ ্ ঠা করার এটা স ্ বীকার করে ।
- EchoRefine Raw: USA Gymnastics supports the United States Olympic Committee's letter and accepts the absolute need of the Olympic family to promote a safe environment for all of our athletes.
- EchoRefine Hybrid: যুক ্ তরাষ ্ ট ্ রের জিমিস ্ ট ্ যাল কম ্ পিটিমের কাছ থেকে সমর ্ থন পাচ ্ ছে এবং অর ্ থনৈতিক প ্ রয়োজন স ্ কুল বাল ্ পীদের জন ্ য একটি স ্ বাচ ্ ছন ্ ন পরিবেশ প ্ রতিষ ্ ঠা করার এটা স ্ বীকার করে ।
- Reference: ইউএসএ জিমন্যাস্টিকস মার্কিন যুক্তরাষ্ট্রের অলিম্পিক কমিটির চিঠিকে সমর্থন করে এবং আমাদের সকল অ্যাথলিটদের ক্ষেত্রে নিরাপদ পরিবেশকে আরও বাড়িয়ে তুলতে অলিম্পিক পরিবারের প্রকৃত প্রয়োজনীয়তার কথা স্বীকার করে।
- Winner: mBART
- Analysis: The raw refinement drifted away from the reference under sentence-level chrF, so the QE gate retained the anchor.

### Anchor Bias Candidate

- Source: The Sundarbans are the largest littoral mangrove belt in the world, stretching 80 km (50 mi) into the Bangladeshi and Indian hinterland from the coast.
- mBART: দন ্ দর ্ বন হচ ্ ছে পৃথিবীর বৃহত ্ তম উপকূলীয় মঙ ্ গর বেল ্ ট, এটা প ্ রদক ্ ষিণ থেকে ৮০ কিলোমিটার (৫০ মাইল) লম ্ বা வங்காளதேச এবং ভারতীয় সাগরতীরে অবস ্ থিত ।
- EchoRefine Raw: সুন্দরবন বিশ্বের বৃহত্তম উপকূলবর্তী ম্যানগ্রোভ বনাঞ্চল, যা ৮০ কিলোমিটার (৫০ মাইল) দীর্ঘ বাংলাদেশ ও ভারতের ভূমির মধ্যে প্�
- EchoRefine Hybrid: দন ্ দর ্ বন হচ ্ ছে পৃথিবীর বৃহত ্ তম উপকূলীয় মঙ ্ গর বেল ্ ট, এটা প ্ রদক ্ ষিণ থেকে ৮০ কিলোমিটার (৫০ মাইল) লম ্ বা வங்காளதேச এবং ভারতীয় সাগরতীরে অবস ্ থিত ।
- Reference: সুন্দরবন বিশ্বের সবথেকে বড় উপকূলবর্তী ম্যানগ্রোভ বন্ধনী, যার পশ্চাজভূমি বাংলাদেশ এবং ভারতের উপকূলে 80 ‌কিমি (50 মাইল) পর্যন্ত প্রসারিত।
- Winner: mBART
- Analysis: The raw refinement scored better than the anchor by sentence-level chrF, but the QE gate retained mBART; this is a candidate anchor-bias case for manual analysis.

## Sinhala (sin)

### Accepted Refinement

- Source: Goma is a tourist city of the Democratic Republic of Congo in the extreme east near Rwanda.
- mBART: goma.com goma.com is a tourist city of the Democratic Republic of Congo in the extreme east near Rwanda.
- EchoRefine Raw: කොමා සංචාරක නගරයක් රවාන්ඩාවේ ඉතා දුරදිගින් පිහිටි කොන්ගෝ ප්රජාතන්ත්‍රවාදී ජනරජයේ �
- EchoRefine Hybrid: කොමා සංචාරක නගරයක් රවාන්ඩාවේ ඉතා දුරදිගින් පිහිටි කොන්ගෝ ප්රජාතන්ත්‍රවාදී ජනරජයේ �
- Reference: ගෝමා යනු රුවන්ඩාව ආසන්නයේ නැගෙනහිර දෙසින් පිහිටි කොංගෝ ප්‍රජාතන්ත්‍රවාදී ජනරජයේ සංචාරක නගරයකි.
- Winner: LLM
- Analysis: The QE gate accepted the MBR-selected refinement; sentence-level chrF suggests the accepted edit moved closer to the reference.

### Overcorrection Prevented

- Source: Their thermal behavior is not as steady as large caves on Earth that often maintain a fairly constant temperature, but it is consistent with these being deep holes in the ground," said Glen Cushing of the United States Geological Survey (USGS) Astrogeology Team and of Northern Arizona University located in Flagstaff, Arizona.
- mBART: ඔවුන්ගෙ උෂ්ණත්වය පෘථිවියෙ ඇති විශාල ගුහාවන් තරම් ස්ථිර නෑ බොහෝ විට සමෘහත උෂ්ණත්වයක් රැකගන්න, නමුත් එය මේ පොළවේ ගැඹුරු හිල් සමග සමීහතයි," ඇරිසෝනා, ෆ් රැග්ස්තාෆ් වල ඇති උතුරු ඇරිසෝනා විශ්ව විද් යාගාරය හා ඇරිසෝනා එක්සත් ජනපද භෞතික විද් යා පරීක්ෂණ (USGS) වායුගෝල විද් යා කණ්ඩායමේ ග්ලේන් කුෂින්ග් කිව්වා.
- EchoRefine Raw: ඔවුන්ගෙ උෂ්ණත්වය පෘථිවියෙ ඇති විශාල ගුහාවන් වගේ නියම නැතිවන, නමුත් ඔවුන්ගෙ උෂ්ණත්
- EchoRefine Hybrid: ඔවුන්ගෙ උෂ්ණත්වය පෘථිවියෙ ඇති විශාල ගුහාවන් තරම් ස්ථිර නෑ බොහෝ විට සමෘහත උෂ්ණත්වයක් රැකගන්න, නමුත් එය මේ පොළවේ ගැඹුරු හිල් සමග සමීහතයි," ඇරිසෝනා, ෆ් රැග්ස්තාෆ් වල ඇති උතුරු ඇරිසෝනා විශ්ව විද් යාගාරය හා ඇරිසෝනා එක්සත් ජනපද භෞතික විද් යා පරීක්ෂණ (USGS) වායුගෝල විද් යා කණ්ඩායමේ ග්ලේන් කුෂින්ග් කිව්වා.
- Reference: ඒවායේ තාප හැසිරීම පෘථිවියේ විශාල ගුහා තරම් ස්ථායී නොවන අතර එය බොහෝ විට තරමක් නියත උෂ්ණත්වයක් පවත්වා ගෙන යන නමුත් එය භූමියේ ගැඹුරු සිදුරු වීම සමග සමපාත වේ ”යැයි  ඇරිසෝනාහි, ෆ්ලැග්ස්ටෆ් හි පිහිටි උතුරු ඇරිසෝනා විශ්ව විද්‍යාලයේ ඇමරිකා එක්සත් ජනපද භූවිද්‍යාත්මක සමීක්ෂණ (USGS) අභ්‍යවකාශ භූ විද්‍යා කණ්ඩායමේ ග්ලෙන් කුෂින් පැවසීය.
- Winner: mBART
- Analysis: The raw refinement drifted away from the reference under sentence-level chrF, so the QE gate retained the anchor.

### Anchor Bias Candidate

- Source: The major religion in Moldova is Orthodox Christian.
- mBART: Moldavia වල ප් රධාන ආගම තමයි Ortodoks Christian.
- EchoRefine Raw: මොල්ඩොවාවේ ප්‍රධාන ආගම තමයි ඔර්තඩෝක්ස් ක්‍රිස්තියානි.
- EchoRefine Hybrid: Moldavia වල ප් රධාන ආගම තමයි Ortodoks Christian.
- Reference: මෝල්දාවේ ප්‍රධාන ආගම වන්නේ සාධර්මික ක්‍රිස්තියානි ආගමයි.
- Winner: mBART
- Analysis: The raw refinement scored better than the anchor by sentence-level chrF, but the QE gate retained mBART; this is a candidate anchor-bias case for manual analysis.

## Burmese (mya)

### Accepted Refinement

- Source: Anyone who's going to drive at high latitudes or over mountain passes should consider the possibility of snow, ice, or freezing temperatures.
- mBART:  anyone who's going to drive at high latitudes or over mountain passes should consider the possibility of snow, ice, or freezing temperatures.
- EchoRefine Raw: မည်းသော အမြင့့်အတိုင်းးများ သို့မဟုတ် တောင်ကြားတောင်တန်းများ ပေါ်သို့ အမြင့့်�
- EchoRefine Hybrid: မည်းသော အမြင့့်အတိုင်းးများ သို့မဟုတ် တောင်ကြားတောင်တန်းများ ပေါ်သို့ အမြင့့်�
- Reference: လတ္တီကျူးအမြင့်များ၌ သို့မဟုတ် တောင်ကြားလမ်းများ၌ မောင်းနှင်ခြင်းကို ပြုလုပ်မည့်သူတိုင်းက နှင်း၊ ရေခဲ သို့မဟုတ် အေးခဲအပူချိန်တို့ ဖြစ်ပေါ်လာနိုင်ခြေကို ထည့်သွင်းစဉ်းစားသင့်သည်။
- Winner: LLM
- Analysis: The QE gate accepted the MBR-selected refinement; sentence-level chrF suggests the accepted edit moved closer to the reference.

### Overcorrection Prevented

- Source: Backpacking by ski: This activity is also called backcountry ski, ski touring or ski hiking.
- mBART: နှင်းလျှောစီး ဖြင့် အပြန်အလှန် ခရီးသွား ခြင်း : ဤ လှုပ်ရှား မှု ကို တောင် နှင်းလျှောစီး ခြင်း ၊ နှင်းလျှောစီး ခရီးသွား ခြင်း သို့မဟုတ် နှင်းလျှောစီး တောင်တက် ခြင်း ဟု လည်း ခေါ် သည် ။
- EchoRefine Raw: နှင်းလျှောစီးနှင်းပါကစ်ချ်ကိုယ်တုံ့ပြီးလိုက်ပါတီးကို ခေါ်ပါသည်။
- EchoRefine Hybrid: နှင်းလျှောစီး ဖြင့် အပြန်အလှန် ခရီးသွား ခြင်း : ဤ လှုပ်ရှား မှု ကို တောင် နှင်းလျှောစီး ခြင်း ၊ နှင်းလျှောစီး ခရီးသွား ခြင်း သို့မဟုတ် နှင်းလျှောစီး တောင်တက် ခြင်း ဟု လည်း ခေါ် သည် ။
- Reference: ကျောပိုးအိတ်နှင့် နှင်းလျှောစီး ခရီးသွားခြင်း- ဤလုပ်ဆောင်ချက်ကို နှင်းလျှောစီးခြင်း၊ နှင်းလျှောစီး၍ ခရီးသွားခြင်း သို့မဟုတ် နှင်းလျှောစီး၍ တောင်တက်ခြင်း ဟုလည်း ခေါ်ဆိုပါသည်။
- Winner: mBART
- Analysis: The raw refinement drifted away from the reference under sentence-level chrF, so the QE gate retained the anchor.

### Anchor Bias Candidate

- Source: As soon as you get out of the current, swimming back is no more difficult than normally.
- mBART: သင် ရေစီးကြောင်း ထဲမှ ထွက် သွား သည် နှင့်တစ်ပြိုင်နက် ၊ ပြန်လည် ရေကူး ခြင်း သည် ပုံမှန် ထက် ပို ၍ ခက်ခဲ ခြင်း မ ရှိ ပါ ။
- EchoRefine Raw: သင်ရေစီးကြောင်းမှထွက်လာသည်နှင့်တစ်ပြိုင်နက် ပြန်လည်ရေကူးခြင်းသည် ပုံမှန်ထက်ပ
- EchoRefine Hybrid: သင် ရေစီးကြောင်း ထဲမှ ထွက် သွား သည် နှင့်တစ်ပြိုင်နက် ၊ ပြန်လည် ရေကူး ခြင်း သည် ပုံမှန် ထက် ပို ၍ ခက်ခဲ ခြင်း မ ရှိ ပါ ။
- Reference: ရေစီးကြောင်းမှ ထွက်လာသည်နှင့် ရေကူးခြင်းသည် ပုံမှန်အားဖြင့် မခက်ခဲတော့ပါ။
- Winner: mBART
- Analysis: The raw refinement scored better than the anchor by sentence-level chrF, but the QE gate retained mBART; this is a candidate anchor-bias case for manual analysis.

## Korean (kor)

### Accepted Refinement

- Source: The Madhya Pradesh Police recovered the stolen laptop and mobile phone.
- mBART: Madhya Pradesh 경찰은 도난된 노트북과 핸드폰을 찾았습니다.
- EchoRefine Raw: 마디야 프라데시 주 경찰은 도난 당한 노트북과 휴대폰을 회수했다.
- EchoRefine Hybrid: 마디야 프라데시 주 경찰은 도난 당한 노트북과 휴대폰을 회수했다.
- Reference: 마디아 프라데시 경찰은 도난당한 노트북과 휴대폰을 회수했다.
- Winner: LLM
- Analysis: The QE gate accepted the MBR-selected refinement; sentence-level chrF suggests the accepted edit moved closer to the reference.

### Overcorrection Prevented

- Source: The truck driver, who is aged 64, was not injured in the crash.
- mBART: 64세의 트럭 운전자는 사고에서 부상을 입지 않았습니다.
- EchoRefine Raw: 사고로 인해 부상자는 없었고, 64세의 운전기사는 사고로 부상이 없었고
- EchoRefine Hybrid: 64세의 트럭 운전자는 사고에서 부상을 입지 않았습니다.
- Reference: 64세인 트럭 운전수는 충돌 사고에서 부상을 입지 않았습니다.
- Winner: mBART
- Analysis: The raw refinement drifted away from the reference under sentence-level chrF, so the QE gate retained the anchor.

### Anchor Bias Candidate

- Source: The major religion in Moldova is Orthodox Christian.
- mBART: Moldova의 주요 종교는 Ortodoks 기독교입니다.
- EchoRefine Raw: 몰도바의 주요 종교는 동방정교회이다.
- EchoRefine Hybrid: Moldova의 주요 종교는 Ortodoks 기독교입니다.
- Reference: 몰도바의 주요 종교는 동방정교입니다.
- Winner: mBART
- Analysis: The raw refinement scored better than the anchor by sentence-level chrF, but the QE gate retained mBART; this is a candidate anchor-bias case for manual analysis.

## Tamil (tam)

### Accepted Refinement

- Source: Sometimes special pyramid shaped towers, called ziggurats, were built to be a part of the temples.
- mBART: sometimes special pyramid shaped towers, called ziggurats, were built to be a part of the temples.
- EchoRefine Raw: சில சமயங்களில், சிறப்பு பிரமிட் வடிவமான கோபுரங்கள், சிக்கரட்டுகள் என்று அழைக்கப்பட்டு, கோயில்களின் ஒரு பகுதியாக க�
- EchoRefine Hybrid: சில சமயங்களில், சிறப்பு பிரமிட் வடிவமான கோபுரங்கள், சிக்கரட்டுகள் என்று அழைக்கப்பட்டு, கோயில்களின் ஒரு பகுதியாக க�
- Reference: சில சமயங்களில் சிறப்பு பிரமிட் வடிவ ஜிக்குராட்டுகள் என்று வழங்கப்பட்ட கோபுரங்களை, கோயில்களின் பாகமாகக் கட்டினார்கள்.
- Winner: LLM
- Analysis: The QE gate accepted the MBR-selected refinement; sentence-level chrF suggests the accepted edit moved closer to the reference.

### Overcorrection Prevented

- Source: Courier companies are well paid for delivering things quickly. Frequently, time is very important with business documents, merchandise or spare parts for an urgent repair.
- mBART: சரக்குகளை விரைவாக விநியோகிப்பதில் கப்பல் நிறுவனங்கள் சிறந்த முறையில் ஊதியம் பெறுகின்றன. அடிக்கடி, வணிக ஆவணங்கள், பொருட்கள் அல்லது அவசரமான பழுதுபார்வைக்கான உபகரணங்கள் ஆகியவற்றில் நேரம் மிகவும் முக்கியமானது.
- EchoRefine Raw: தந்தையின் திருமணம்
- EchoRefine Hybrid: சரக்குகளை விரைவாக விநியோகிப்பதில் கப்பல் நிறுவனங்கள் சிறந்த முறையில் ஊதியம் பெறுகின்றன. அடிக்கடி, வணிக ஆவணங்கள், பொருட்கள் அல்லது அவசரமான பழுதுபார்வைக்கான உபகரணங்கள் ஆகியவற்றில் நேரம் மிகவும் முக்கியமானது.
- Reference: பொருட்களை வேகமாக வழங்குவதற்கு கூரியர் நிறுவங்கள் நல்ல வருமானங்களை பெறுகின்றன. அடிகடி வணிக ஆவணங்கள், பொருட்கள் அல்லது அவசர பழுதுபார்க்க உதிரி பாகங்களுடன் நேரம் மிகவும் முக்கியமானது.
- Winner: mBART
- Analysis: The raw refinement drifted away from the reference under sentence-level chrF, so the QE gate retained the anchor.

### Anchor Bias Candidate

- Source: The first public reading of the document was by John Nixon in the yard of Independence Hall on July 8.
- mBART: ஜூலை 8 அன்று ஜான் நிக்சன் சுயாதீன மண்டபத்தின் மைதானத்தில் இந்த ஆவணத்தை பகிரங்கமாக வாசித்தார்.
- EchoRefine Raw: இந்த ஆவணத்தின் முதல் பொது வாசிப்பு ஜூலை 8 அன்று சுதந்திர மண்டபத்தின் வளாகத்தில் ஜான் நிக்சன் ஆல் செய்யப்பட்டது.
- EchoRefine Hybrid: ஜூலை 8 அன்று ஜான் நிக்சன் சுயாதீன மண்டபத்தின் மைதானத்தில் இந்த ஆவணத்தை பகிரங்கமாக வாசித்தார்.
- Reference: இந்த ஆவணத்தின் முதல் பொது வாசிப்பு ஜூலை 8 அன்று சுதந்திர மண்டபத்தின் முற்றத்தில் ஜான் நிக்சன் எழுதியது.
- Winner: mBART
- Analysis: The raw refinement scored better than the anchor by sentence-level chrF, but the QE gate retained mBART; this is a candidate anchor-bias case for manual analysis.

## Hindi (hin)

### Accepted Refinement

- Source: The photographer was transported to Ronald Reagan UCLA Medical Center, where he subsequently died.
- mBART: फोटोग्राफर को Ronald Reagan UCLA Medical Center ले जाया गया, जहां वह बाद में मर गया।
- EchoRefine Raw: फोटोग्राफर को रोनाल्ड रीगन यूसीएलए मेडिकल सेंटर ले जाया गया, जहां बाद में उसकी मृत्यु हो गई।
- EchoRefine Hybrid: फोटोग्राफर को रोनाल्ड रीगन यूसीएलए मेडिकल सेंटर ले जाया गया, जहां बाद में उसकी मृत्यु हो गई।
- Reference: फ़ोटोग्राफ़र को रोनाल्ड रीगन यूसीएलए मेडिकल सेंटर ले जाया गया, जहाँ बाद में उसकी मौत हो गई.
- Winner: LLM
- Analysis: The QE gate accepted the MBR-selected refinement; sentence-level chrF suggests the accepted edit moved closer to the reference.

### Overcorrection Prevented

- Source: As knowledge of Greek declined, the West found itself cut off from its Greek philosophical and scientific roots.
- mBART: जैसे-जैसे ग्रीक का ज्ञान घटता गया, पश्चिम अपने ग्रीक दार्शनिक और वैज्ञानिक मूलों से अलग हो गया।
- EchoRefine Raw: जैसे ही ग्रीक के ज्ञान में कमी आई, पश्चिम अपने ग्रीक दर्शनशास्त्र और विज्ञान के मूलों से अलग हो गया.
- EchoRefine Hybrid: जैसे-जैसे ग्रीक का ज्ञान घटता गया, पश्चिम अपने ग्रीक दार्शनिक और वैज्ञानिक मूलों से अलग हो गया।
- Reference: जैसे-जैसे ग्रीक का ज्ञान घटता गया, पश्चिम ने खुद को अपनी ग्रीक दार्शनिक और वैज्ञानिक जड़ों से कटा हुआ पाया.
- Winner: mBART
- Analysis: The raw refinement drifted away from the reference under sentence-level chrF, so the QE gate retained the anchor.

### Anchor Bias Candidate

- Source: Vatican City uses Italian in its legislation and official communications.
- mBART: वैटीकन सिटी अपने विधानों और सरकारी संचारों में इतालवी भाषा का प्रयोग करता है।
- EchoRefine Raw: वैटिकन सिटी अपने कानून और आधिकारिक संचार में इतालवी का उपयोग करता है.
- EchoRefine Hybrid: वैटीकन सिटी अपने विधानों और सरकारी संचारों में इतालवी भाषा का प्रयोग करता है।
- Reference: वैटिकन सिटी अपने कानूनी और आधिकारिक संचार में इतालवी का उपयोग करता है।
- Winner: mBART
- Analysis: The raw refinement scored better than the anchor by sentence-level chrF, but the QE gate retained mBART; this is a candidate anchor-bias case for manual analysis.

## French (fra)

### Accepted Refinement

- Source: Ocelots like to eat small animals. They will catch monkeys, snakes, rodents and birds if they can. Almost all of the animals that the ocelot hunts are far smaller than it is.
- mBART: Les océlots aiment manger de petits animaux, ils capturent les apes, les serpents, les rongeurs et les oiseaux s'ils peuvent.
- EchoRefine Raw: Les ocelots aiment manger de petits animaux. Ils attraperont des singes, des serpents, des rongeurs et des oiseaux s'ils le peuvent. Presque tous les animaux que l'ocelot chasse sont bien plus petits qu'il.
- EchoRefine Hybrid: Les ocelots aiment manger de petits animaux. Ils attraperont des singes, des serpents, des rongeurs et des oiseaux s'ils le peuvent. Presque tous les animaux que l'ocelot chasse sont bien plus petits qu'il.
- Reference: Les ocelots sont friands de petits animaux. Ils attrapent des singes, des serpents, des rongeurs et des oiseaux s'ils le peuvent. Presque tous les animaux que l'ocelot chasse sont bien plus petits que lui.
- Winner: LLM
- Analysis: The QE gate accepted the MBR-selected refinement; sentence-level chrF suggests the accepted edit moved closer to the reference.

### Overcorrection Prevented

- Source: You can see the pyramids in the dark and you can see them in silence before the show begins.
- mBART: Vous pouvez voir les pyramides dans l'obscurité et vous pouvez les voir en silence avant le début du spectacle.
- EchoRefine Raw: On peut voir les pyramides dans le noir, et on peut les voir dans le silence avant que le spectacle ne commence.
- EchoRefine Hybrid: Vous pouvez voir les pyramides dans l'obscurité et vous pouvez les voir en silence avant le début du spectacle.
- Reference: Vous pouvez voir les pyramides dans l'obscurité et vous pouvez les voir en silence avant le début du spectacle.
- Winner: mBART
- Analysis: The raw refinement drifted away from the reference under sentence-level chrF, so the QE gate retained the anchor.

### Anchor Bias Candidate

- Source: It takes a giant rocket over a 100 feet high to put a satellite or telescope in space.
- mBART: Il faut une roquette géante à plus de 100 pieds de hauteur pour placer un satellite ou un télescope dans l'espace.
- EchoRefine Raw: Il faut une fusée géante de plus de 30 mètres de haut pour mettre un satellite ou un télescope dans l'espace.
- EchoRefine Hybrid: Il faut une roquette géante à plus de 100 pieds de hauteur pour placer un satellite ou un télescope dans l'espace.
- Reference: Il faut une fusée géante de plus de 30 mètres de haut pour mettre un satellite ou un télescope dans l’espace.
- Winner: mBART
- Analysis: The raw refinement scored better than the anchor by sentence-level chrF, but the QE gate retained mBART; this is a candidate anchor-bias case for manual analysis.
